"""Locate and (optionally) supervise a user-local Ollama daemon.

Design doc §4.1/§N5: backends are external processes. By default Hearth
*discovers* a running Ollama; with supervision on (the bundled-binary case,
which is our install path here) it starts ``ollama serve`` itself and stops
it on exit, so a clean ``hearth start`` brings up the whole stack.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

# Where our user-local install puts the binary (vendor/ollama/bin/ollama).
_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "ollama" / "bin" / "ollama"


def ollama_binary() -> str | None:
    env = os.environ.get("HEARTH_OLLAMA_BIN")
    if env and Path(env).exists():
        return env
    if _VENDOR.exists():
        return str(_VENDOR)
    from shutil import which

    return which("ollama")


def is_up(base_url: str, timeout: float = 1.5) -> bool:
    try:
        return httpx.get(f"{base_url.rstrip('/')}/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def start(base_url: str, wait_s: float = 30.0) -> subprocess.Popen | None:
    """Start ``ollama serve`` if it isn't already up. Returns the process we
    started (so the caller can stop it), or None if Ollama was already up or
    no binary is available."""
    if is_up(base_url):
        return None
    binary = ollama_binary()
    if not binary:
        return None
    # OLLAMA_HOST controls the daemon's bind. Mirror our configured base_url.
    host = base_url.split("://", 1)[-1]
    env = {**os.environ, "OLLAMA_HOST": host}
    proc = subprocess.Popen(
        [binary, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    def _stop():
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()

    atexit.register(_stop)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if is_up(base_url):
            return proc
        if proc.poll() is not None:
            return None  # died on startup
        time.sleep(0.5)
    return proc  # may still be warming; caller surfaces health
