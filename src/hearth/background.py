"""Start the gateway so that it outlives whoever asked.

`hearth start --background` is what an app uses (finterm, the chat window): it
returns once the gateway answers, and the gateway stays when the app is gone.
Where hearth is installed as a service, the service is what is started, so the
engine keeps one owner.
"""

from __future__ import annotations

import os
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
    duplicate `hearth start` then concludes the port is free, announces itself,
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


def _last_lines(since: int, count: int = 3) -> str:
    """The end of what was logged from byte `since` on."""
    try:
        with open(log_path(), "rb") as f:
            f.seek(since)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join([ln for ln in text.splitlines() if ln.strip()][-count:])


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
                      cwd=str(Path.home()))
        try:
            return subprocess.Popen(argv, **common, **kwargs), before
        except OSError:
            if os.name == "posix":
                raise
            kwargs["creationflags"] &= ~0x01000000  # a job that allows no leaving
            return subprocess.Popen(argv, **common, **kwargs), before


def start(cfg: cfgmod.Config, hearth_exec: list[str], extra: list[str] | None = None,
          start_service=None, wait_s: float = _WAIT_S) -> tuple[bool, str]:
    """(serving, what to tell the user). Returns when the gateway answers, its
    process ends, or wait_s is over. `start_service` starts the installed
    service and returns its exit code; without it the gateway is spawned."""
    if (ident := gateway_identity(cfg)) is not None:
        return True, f"already running ({ident})"
    proc, before = None, 0
    if start_service is not None:
        if (rc := start_service()) != 0:
            return False, f"the hearth service did not start (exit {rc}) — see `hearth service logs`"
    else:
        try:
            proc, before = spawn([*hearth_exec, "start", *(extra or [])])
        except OSError as e:
            return False, f"could not run {hearth_exec[0] if hearth_exec else 'hearth'}: {e}"
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if (ident := gateway_identity(cfg)) is not None:
            return True, f"running ({ident})"
        if proc is not None and proc.poll() is not None:
            why = _last_lines(before)
            return False, f"hearth ended (exit {proc.returncode})" + (f":\n{why}" if why else "")
        time.sleep(0.25)
    where = "`hearth service logs`" if start_service is not None else str(log_path())
    return False, f"hearth did not answer in {wait_s:.0f} s — see {where}"
