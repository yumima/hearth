"""Locate, run and keep an eye on Ollama.

Design doc §4.1/§N5: backends are external processes. An Ollama that already
answers at the configured address is used and left alone. Otherwise hearth
starts one, keeps its output, starts it again when it dies, and ends it on exit.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from shutil import which
from urllib.parse import urlsplit

import httpx

from . import paths, procs

log = logging.getLogger("hearth.ollama")

# The context window we start Ollama with (see _env()). Exported because the
# gateway has to size tool results against it: Ollama silently drops the OLDEST
# messages when a request overruns the window, so an oversized fetched page
# doesn't error — it quietly deletes the system prompt and the user's question.
DEFAULT_CONTEXT_LENGTH = 8192

_LOG_MAX = 5_000_000
_LOOK_S = 3.0             # how often the watcher looks
_BACKOFF_S = (1, 2, 5, 15, 60)
_SETTLED_S = 300.0        # up this long, and the next failure is a new one


def effective_default_context() -> int:
    """The daemon-wide window for requests with no context of their own: what
    start() launches with, an operator's OLLAMA_CONTEXT_LENGTH included."""
    try:
        value = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", ""))
    except ValueError:
        return DEFAULT_CONTEXT_LENGTH
    return value if value > 0 else DEFAULT_CONTEXT_LENGTH


def ollama_binary() -> str | None:
    """HEARTH_OLLAMA_BIN, the installed one, a source checkout's vendor/, PATH."""
    env = os.environ.get("HEARTH_OLLAMA_BIN")
    if env and Path(env).exists():
        return env
    exe = "ollama.exe" if sys.platform == "win32" else "ollama"
    for root in (paths.data_dir() / "ollama", paths.find_in_checkout("vendor", "ollama")):
        for rel in ("bin", ""):  # the Linux archive has bin/, the others do not
            if root is not None and (p := root / rel / exe).is_file():
                return str(p)
    return which("ollama")


def is_up(base_url: str, timeout: float = 1.5) -> bool:
    try:
        return httpx.get(f"{base_url.rstrip('/')}/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def log_path() -> Path:
    return paths.home() / "ollama.log"


def _pid_path() -> Path:
    return paths.home() / "ollama.pid"


def _env(base_url: str, models_dir: str) -> dict:
    # OLLAMA_HOST controls the daemon's bind. Mirror our configured base_url.
    env = {**os.environ, "OLLAMA_HOST": urlsplit(base_url).netloc}
    if models_dir:
        env["OLLAMA_MODELS"] = models_dir
    # Raise the default model context window above Ollama's 4096. Tool-heavy chat
    # (curated MCP tool schemas + history + tool results) overruns 4096 and the
    # model loses the tools — which broke tool-calling. This is only the floor for
    # requests with no context of their own: a role's bound context overrides it,
    # sized per model so the KV cache stays in VRAM (hardware.recommend_contexts).
    # Respect an explicit operator override if one is already set.
    env.setdefault("OLLAMA_CONTEXT_LENGTH", str(DEFAULT_CONTEXT_LENGTH))
    # Quantize the K/V cache to 8-bit. The cache scales linearly with context and
    # at 32k it outweighs the model weights themselves; q8_0 halves it for a
    # perplexity cost in the 0.002-0.05 range — i.e. free context on a 12 GB card.
    # It only takes effect with flash attention on (Ollama silently falls back to
    # f16 otherwise, which reads as an unexplained OOM), so the two are set together.
    env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    env.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
    # Keep the chat model resident longer than Ollama's 5-minute default so a brief
    # pause between messages doesn't trigger a ~15-20s cold reload. (Image-gen still
    # evicts explicitly via keep_alive:0, then re-warms — see routes/v1.py.)
    env.setdefault("OLLAMA_KEEP_ALIVE", os.environ.get("HEARTH_KEEP_ALIVE", "30m"))
    return env


class Supervisor:
    """One Ollama, ours. start() → (kept up by a thread) → stop()."""

    def __init__(self, base_url: str, models_dir: str = ""):
        self.base_url = base_url
        self.models_dir = models_dir
        self._pid = 0
        self._since = 0.0
        self._failures = 0
        self._stopping = threading.Event()
        self._lock = threading.Lock()  # a start and the stop do not cross
        self._proc: subprocess.Popen | None = None

    # ── one start ────────────────────────────────────────────────────────────
    def _spawn(self) -> bool:
        binary = ollama_binary()
        if not binary:
            return False
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            if p.stat().st_size > _LOG_MAX:
                p.replace(p.with_suffix(".log.1"))
        except OSError:
            pass
        with self._lock:
            if self._stopping.is_set():
                return False
            try:
                with open(p, "ab") as out:
                    self._proc = subprocess.Popen(
                        [binary, "serve"], env=_env(self.base_url, self.models_dir),
                        stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **procs.own_group())
            except OSError as e:
                log.error("could not run %s: %s", binary, e)
                return False
            self._own(self._proc.pid)
        return True

    def _own(self, pid: int) -> None:
        self._pid, self._since = pid, time.monotonic()
        me = os.getpid()
        try:
            _pid_path().write_text(json.dumps({
                "pid": pid, "started": procs.started_at(pid), "base_url": self.base_url,
                "owner": me, "owner_started": procs.started_at(me)}))
        except OSError:
            pass

    def _ended(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is not None
        return not procs.alive(self._pid)

    def _adopt(self) -> bool:
        """One that a hearth which was killed had started, at this address: ours
        to end, still. Not one whose hearth lives (a second gateway of the same
        home), nor one a pid only looks like."""
        try:
            was = json.loads(_pid_path().read_text())
            pid, started, owner, owner_started = (
                int(was["pid"]), float(was["started"]), int(was["owner"]), float(was["owner_started"]))
            if was["base_url"] != self.base_url:
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if not started or procs.started_at(pid) != started or "ollama" not in procs.command_line(pid).lower():
            return False
        if owner != os.getpid() and owner_started and procs.started_at(owner) == owner_started:
            return False
        self._own(pid)
        return True

    def start(self, wait_s: float = 30.0, keep_up: bool = True) -> bool:
        """Is Ollama ours now? False: one was already up that is not ours, or
        there is none to run (is_up() tells which). `keep_up`: start it again
        when it ends, and end it on exit — a gateway's. Without, a command's for
        as long as it runs."""
        if is_up(self.base_url):
            if not self._adopt():
                return False
            log.info("took over the Ollama of an earlier run (pid %s)", self._pid)
        elif not self._spawn():
            return False
        atexit.register(self.stop)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not is_up(self.base_url):
            if self._ended():
                break  # started again below; what it said is in the log
            time.sleep(0.5)
        if keep_up:
            threading.Thread(target=self._watch, name="ollama-watch", daemon=True).start()
        return True

    # ── staying up ───────────────────────────────────────────────────────────
    def _watch(self) -> None:
        while not self._stopping.wait(_LOOK_S):
            if not self._ended():
                continue
            if time.monotonic() - self._since > _SETTLED_S:
                self._failures = 0
            wait = _BACKOFF_S[min(self._failures, len(_BACKOFF_S) - 1)]
            self._failures += 1
            log.error("Ollama ended (exit %s) — starting it again in %s s; its output is in %s",
                      self._proc.returncode if self._proc else "?", wait, log_path())
            if self._stopping.wait(wait):
                return
            # Somebody else's may have the address now (a command's own, say).
            # It is used while it is there, and ours comes back when it goes.
            while is_up(self.base_url):
                if self._stopping.wait(_LOOK_S):
                    return
            self._spawn()

    def stop_watching(self) -> None:
        """The gateway is going down: what ends now stays ended."""
        self._stopping.set()

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            pid, self._pid = self._pid, 0
            if not pid:
                return
            if self._proc is not None or procs.alive(pid):
                procs.end_tree(pid)
            try:
                if json.loads(_pid_path().read_text()).get("pid") == pid:
                    _pid_path().unlink()
            except (OSError, ValueError, AttributeError):
                pass


def last_lines(count: int = 8) -> str:
    """The end of Ollama's own output, for a person who wants to know why."""
    try:
        with open(log_path(), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-count:])
    except OSError:
        return ""
