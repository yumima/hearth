"""Start the gateway so that it outlives whoever asked.

`hearth start` is what an app uses, too (finterm, the chat window): it
returns once the gateway answers, and the gateway stays when the app is gone.
Where hearth is installed as a service, the service is what is started, so the
engine keeps one owner. (A caller that is itself a systemd service takes what it
spawns down with it: there, install hearth's service.)
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import httpx

from . import config as cfgmod

_LOG_MAX = 5_000_000  # then it is moved aside, once
_WAIT_S = 45.0        # importing the server and bringing up Ollama, on a slow disk


def gateway_identity(cfg: cfgmod.Config) -> str | None:
    """`Server:` identity of a hearth gateway already serving cfg's port, else None.

    Probes /admin/version — a constant handler — and deliberately NOT
    /admin/health: health awaits every configured backend, remote providers
    included, so one slow round-trip over the internet times the probe out. A
    duplicate `hearth serve` then concludes the port is free, announces itself,
    and dies on EADDRINUSE.
    """
    try:
        r = httpx.get(f"http://{cfg.bind_host}:{cfg.bind_port}/admin/version", timeout=2.0)
    except httpx.HTTPError:
        return None
    ident = r.headers.get("server", "")
    if ident.startswith("hearth"):
        return ident
    try:
        body = r.json()
    except ValueError:
        return None
    if isinstance(body, dict) and body.get("name") == "hearth":
        return f"hearth/{body.get('version', '?')}"
    return None


def _coming_up(cfg: cfgmod.Config) -> bool:
    """Does the port take a connection and say nothing? A hearth listens before
    it brings up Ollama and answers after. Anything else on the port answers
    (or refuses) at once."""
    try:
        with socket.create_connection((cfg.bind_host, cfg.bind_port), timeout=1.0) as s:
            s.settimeout(0.5)
            s.sendall(b"GET /admin/version HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            try:
                s.recv(1)
                return False
            except TimeoutError:
                return True
    except OSError:
        return False


def log_path() -> Path:
    return cfgmod.config_path().parent / "hearth.log"


def _open_log():
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if p.stat().st_size > _LOG_MAX:
            p.replace(p.with_suffix(".log.1"))
    except OSError:
        pass
    return open(p, "ab")


def _last_lines(since: int, count: int = 6) -> str:
    """The end of what was logged from byte `since` on."""
    try:
        with open(log_path(), "rb") as f:
            f.seek(since)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join([ln for ln in text.splitlines() if ln.strip()][-count:])


def _child_env() -> dict:
    """Ours, for a process in another folder: what names a file still names it."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # the log in order, whatever ends it
    for name in ("HEARTH_HOME", "HEARTH_CONFIG", "HEARTH_DATA", "HEARTH_KOKORO_DIR", "HEARTH_OLLAMA_BIN"):
        if env.get(name):
            env[name] = os.path.abspath(os.path.expanduser(env[name]))
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.abspath(p) if p else "" for p in env["PYTHONPATH"].split(os.pathsep))
    return env


def spawn(argv: list[str]) -> tuple[subprocess.Popen, int]:
    """Run argv on its own, its output in the log; (process, log size before)."""
    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # a closing terminal or app does not reach it
    else:
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB:
        # out of the job an app may have put its helpers in, to end with it
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x01000000
    with _open_log() as log:
        before = log.tell()
        common = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                      cwd=str(log_path().parent), env=_child_env())
        try:
            return subprocess.Popen(argv, **common, **kwargs), before
        except OSError:
            if os.name == "posix":
                raise
            kwargs["creationflags"] &= ~0x01000000  # a job that allows no leaving
            return subprocess.Popen(argv, **common, **kwargs), before


class Service:
    """The installed service: how to start it, and whether it is still trying."""

    def __init__(self, start, alive):
        self.start = start  # () -> exit code
        self.alive = alive  # () -> bool


def start(cfg: cfgmod.Config, argv: list[str], service: Service | None = None,
          wait_s: float = _WAIT_S) -> tuple[bool, str]:
    """(serving, what to tell the user). Returns when the gateway answers, when
    what was started has ended and nothing is coming up, or after wait_s."""
    if (ident := gateway_identity(cfg)) is not None:
        return True, f"already running ({ident})"
    proc, before = None, 0
    if service is not None:
        if (rc := service.start()) != 0:
            return False, f"the hearth service did not start (exit {rc}) — see `hearth logs`"
    else:
        try:
            proc, before = spawn(argv)
        except OSError as e:
            return False, f"could not run {argv[0] if argv else 'hearth'}: {e}"
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if (ident := gateway_identity(cfg)) is not None:
            return True, f"running ({ident})"
        ended = proc.poll() is not None if proc is not None else not service.alive()
        # It may have ended because another one is on its way up: that one counts.
        if ended and not _coming_up(cfg):
            if (ident := gateway_identity(cfg)) is not None:  # up since we looked
                return True, f"running ({ident})"
            if proc is None:
                return False, "the hearth service ended — see `hearth logs`"
            why = _last_lines(before)
            return False, f"hearth ended (exit {proc.returncode})" + (f":\n{why}" if why else "")
        time.sleep(0.25)
    where = "`hearth logs`" if service is not None else str(log_path())
    return False, f"hearth did not answer in {wait_s:.0f} s — see {where}"
