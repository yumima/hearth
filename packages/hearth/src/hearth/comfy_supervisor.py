"""On-demand ComfyUI supervisor.

ComfyUI (the SDXL backend for /v1/images/generations) idles at ~6 GB RAM. On a
memory-tight box that is a standing contribution to swap-thrash freezes, even
when no images are being generated. So hearth runs it *on demand*: start it when
an image is requested, then stop it after an idle period to give the RAM back.

Ownership rule: hearth only stops a ComfyUI it started itself. If one is already
running (an operator launched it), hearth uses it and never touches its lifetime.

All the async entry points share hearth's single event loop, so plain module
state guarded by one asyncio.Lock is sufficient — no threads involved.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
# Blackwell (sm_120) produces black/NaN images in fp16; --bf16-unet --fp32-vae
# fixes that. --highvram keeps the model resident across prompts within a session.
_ARGS = os.environ.get("HEARTH_COMFY_ARGS", "--bf16-unet --fp32-vae --highvram")
_IDLE = float(os.environ.get("HEARTH_COMFY_IDLE_TIMEOUT", "300"))   # stop after Ns idle
_START_WAIT = float(os.environ.get("HEARTH_COMFY_START_WAIT", "120"))  # readiness timeout

# Set only when WE started ComfyUI (so we never stop an operator-run instance).
_proc: subprocess.Popen | None = None
_lock = asyncio.Lock()
_last_use = 0.0
_active = 0                 # in-flight generations — never idle-stop while > 0
_idle_task: asyncio.Task | None = None


def _state_log() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "hearth" / "comfyui.log"


def _resolve_dir() -> Path | None:
    """Locate a ComfyUI checkout (one containing main.py)."""
    env = os.environ.get("HEARTH_COMFY_DIR")
    candidates = [env] if env else []
    # Sibling of the hearth repo (…/fin/comfyui), then common spots.
    candidates += [
        str(Path(__file__).resolve().parents[3] / "comfyui"),
        str(Path.home() / "comfyui"),
        str(Path.home() / "ComfyUI"),
        str(Path.home() / "fin" / "comfyui"),
    ]
    for c in candidates:
        if c and (Path(c) / "main.py").exists():
            return Path(c)
    return None


def _resolve_python(d: Path) -> str:
    env = os.environ.get("HEARTH_COMFY_PYTHON")
    if env:
        return env
    for c in (d / "venv" / "bin" / "python", d / ".venv" / "bin" / "python"):
        if c.exists():
            return str(c)
    return "python3"


async def _is_up(timeout: float = 2.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            return (await c.get(f"{_URL}/system_stats")).status_code == 200
    except httpx.HTTPError:
        return False


def _touch() -> None:
    global _last_use
    _last_use = time.monotonic()


async def ensure_up() -> bool:
    """Make ComfyUI reachable, starting it if hearth can. Returns False only when
    it isn't up and we have no way to start it (no checkout found / it died)."""
    _touch()
    if await _is_up():
        _ensure_idle_watcher()
        return True
    async with _lock:
        if await _is_up():            # someone won the race while we waited
            _ensure_idle_watcher()
            return True
        d = _resolve_dir()
        if d is None:
            return False
        py = _resolve_python(d)
        port = urlparse(_URL).port or 8188
        cmd = [py, "main.py", "--port", str(port), *shlex.split(_ARGS)]
        log = _open_log()
        out = log or subprocess.DEVNULL
        global _proc
        try:
            _proc = subprocess.Popen(
                cmd, cwd=str(d), start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        except OSError:
            _proc = None
            if log:
                log.close()
            return False
        if log:
            log.close()  # child inherited the fd
        # Wait for readiness — torch + ComfyUI import is slow (cold ~10-30s).
        deadline = time.monotonic() + _START_WAIT
        while time.monotonic() < deadline:
            if _proc.poll() is not None:   # died during startup
                _proc = None
                return False
            if await _is_up():
                _touch()
                _ensure_idle_watcher()
                return True
            await asyncio.sleep(1.0)
        return await _is_up()


def begin() -> None:
    """Mark a generation in flight (suppresses idle-stop) and refresh last-use."""
    global _active
    _active += 1
    _touch()


def end() -> None:
    global _active
    _active = max(0, _active - 1)
    _touch()


def _open_log():
    lp = _state_log()
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        if lp.exists() and lp.stat().st_size > 2_000_000:
            lp.replace(lp.with_name(lp.name + ".1"))
        return open(lp, "a", buffering=1, encoding="utf-8")
    except OSError:
        return None


def _ensure_idle_watcher() -> None:
    global _idle_task
    if _proc is None:                 # we don't own it → don't police its lifetime
        return
    if _idle_task is None or _idle_task.done():
        _idle_task = asyncio.create_task(_idle_loop())


async def _idle_loop() -> None:
    global _proc, _idle_task
    try:
        while _proc is not None:
            await asyncio.sleep(min(_IDLE, 30.0))
            if _proc is None:
                return
            if _proc.poll() is not None:      # exited on its own
                _proc = None
                return
            if _active == 0 and (time.monotonic() - _last_use) >= _IDLE:
                # Serialize the stop with ensure_up() and re-check under the lock:
                # a request that slipped in after the idle check (bumping _active or
                # _last_use) must not be cut off, and no new start can interleave
                # with the kill.
                async with _lock:
                    if _active == 0 and (time.monotonic() - _last_use) >= _IDLE:
                        await _stop()
                        return
                # else: activity resumed — keep watching
    finally:
        _idle_task = None


async def _stop() -> None:
    global _proc
    p, _proc = _proc, None
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            p.terminate()
        except OSError:
            return
    for _ in range(50):                # up to ~10s for a graceful exit
        if p.poll() is not None:
            return
        await asyncio.sleep(0.2)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def shutdown() -> None:
    """Synchronous best-effort stop for atexit — kill the ComfyUI we started so it
    doesn't outlive hearth."""
    p = _proc
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


atexit.register(shutdown)
