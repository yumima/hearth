"""hearth's own SearXNG, the default web-search backend.

Without it — or a Brave key — search was down to Brave's HTML page, which now
answers 429 on the first request, and Wikipedia. SearXNG is the one keyless
route that holds up (it reaches Google's index too; see websearch.py), so the
installer puts one in every installation: versions/V/searxng, built from the
release's pinned commit and hash-checked like the rest. No container, no
account, no question.

It runs on demand, like ComfyUI (comfy_supervisor.py): the first search starts
it on a free loopback port, and it is stopped after an idle spell to give its
memory back. A `search.searxng_url` someone set is theirs and is used instead —
this one is only for when there is none.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import re
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from . import __version__, paths, procs

PORTS = range(8888, 8899)
_START_WAIT = 45.0   # cold start, its engines loaded: ~5 s here
_IDLE = 1800.0       # an agent session searches in bursts; half an hour covers one
_RETRY = 600.0       # after a start that failed: not every search waits for another

# use_default_settings keeps upstream's engines; this changes only what hearth
# needs: loopback, the JSON API (websearch._searxng reads it), and no bot
# limiter — the only client is this machine, and the limiter wants a Valkey.
SETTINGS = """\
# Made by hearth for the SearXNG it runs; it is remade on each start.
use_default_settings: true
server:
  secret_key: "{secret}"
  bind_address: "127.0.0.1"
  port: {port}
  limiter: false
  image_proxy: false
search:
  formats:
    - html
    - json
"""

_proc: subprocess.Popen | None = None
_url = ""
_lock = asyncio.Lock()
_last_use = 0.0
_idle_task: asyncio.Task | None = None
_failed_at = -_RETRY


def env() -> Path:
    return paths.data_dir() / "versions" / __version__ / "searxng"


def python() -> Path | None:
    """The installed environment's Python, if this installation has one."""
    e = env()
    if not (e / ".complete").is_file():
        return None
    p = e / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return p if p.is_file() else None


def folder() -> Path:
    return paths.data_dir() / "searxng"


def write_settings(where: Path, port: int) -> Path:
    """settings.yml for this start; the secret of the last one is kept."""
    where.mkdir(parents=True, exist_ok=True)
    p = where / "settings.yml"
    old = p.read_text(encoding="utf-8") if p.exists() else ""
    m = re.search(r'secret_key: "([0-9a-f]{64})"', old)
    p.write_text(SETTINGS.format(secret=m.group(1) if m else secrets.token_hex(32), port=port), encoding="utf-8")
    return p


def _free_port() -> int | None:
    for port in PORTS:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


async def _answers(url: str, timeout: float = 2.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            return (await c.get(f"{url}/healthz")).status_code == 200
    except httpx.HTTPError:
        return False


def _open_log():
    lp = paths.home() / "searxng.log"
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        if lp.exists() and lp.stat().st_size > 2_000_000:
            lp.replace(lp.with_name(lp.name + ".1"))
        return open(lp, "a", buffering=1, encoding="utf-8")
    except OSError:
        return None


async def ensure_up() -> str:
    """The URL of a running SearXNG of ours, started if need be; "" when this
    installation has none or it would not start (search goes on without it)."""
    global _proc, _url, _last_use, _failed_at
    _last_use = time.monotonic()
    if _proc is not None and _proc.poll() is None and await _answers(_url):
        return _url
    async with _lock:
        if _proc is not None and _proc.poll() is None and await _answers(_url):
            return _url
        await _stop()
        if time.monotonic() - _failed_at < _RETRY:
            return ""
        py, port = python(), _free_port()
        if py is None or port is None:
            return ""
        _failed_at = time.monotonic()  # until it answers
        where = folder()
        settings = write_settings(where, port)
        log = _open_log()
        try:
            _proc = subprocess.Popen(
                [str(py), "-I", "-m", "searx.webapp"], cwd=str(where), **procs.own_group(),
                env={**os.environ, "SEARXNG_SETTINGS_PATH": str(settings)},
                stdin=subprocess.DEVNULL, stdout=log or subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except OSError:
            _proc = None
            return ""
        finally:
            if log:
                log.close()  # the child has it
        _url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + _START_WAIT
        while time.monotonic() < deadline:
            if _proc.poll() is not None:
                _proc = None
                return ""
            if await _answers(_url):
                _failed_at = -_RETRY
                _ensure_idle_watcher()
                return _url
            await asyncio.sleep(0.5)
        await _stop()
        return ""


def _ensure_idle_watcher() -> None:
    global _idle_task
    if _idle_task is None or _idle_task.done():
        _idle_task = asyncio.create_task(_idle_loop())


async def _idle_loop() -> None:
    while _proc is not None and _proc.poll() is None:
        await asyncio.sleep(min(_IDLE, 60.0))
        if time.monotonic() - _last_use >= _IDLE:
            async with _lock:
                if time.monotonic() - _last_use >= _IDLE:
                    await _stop()
                    return


async def _stop() -> None:
    global _proc
    p, _proc = _proc, None
    if p is not None and p.poll() is None:
        await asyncio.to_thread(procs.end_tree, p.pid)


def shutdown() -> None:
    """Best-effort, for atexit: it does not outlive hearth."""
    p = _proc
    if p is not None and p.poll() is None:
        procs.end_tree(p.pid, wait_s=3)


atexit.register(shutdown)
