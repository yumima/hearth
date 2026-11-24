"""Other processes, the same way on Linux, macOS and Windows."""

from __future__ import annotations

import subprocess
import sys

import psutil


def alive(pid: int) -> bool:
    """(Not os.kill(pid, 0): on Windows that ends the process.)"""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True  # there, and not ours to look at


def command_line(pid: int) -> str:
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except psutil.Error:
        return ""


def started_at(pid: int) -> float:
    """When it started, 0 if it is gone: a pid alone may be another process by now."""
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return 0.0


def listening_on(port: int) -> int | None:
    """The process listening on a TCP port, if we may see it (our own user's)."""
    def match(conns):
        return any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port for c in conns)

    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.pid and match([c]):
                return c.pid
        return None
    except psutil.Error:
        pass  # macOS lists every process's connections only to root
    for p in psutil.process_iter():
        try:
            if match(p.net_connections(kind="tcp")):
                return p.pid
        except psutil.Error:
            continue
    return None


def end_tree(pid: int, wait_s: float = 10.0) -> None:
    """End a process and what it started: asked first, then made to."""
    try:
        top = psutil.Process(pid)
        procs = [top, *top.children(recursive=True)]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, left = psutil.wait_procs(procs, timeout=wait_s)
    for p in left:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(left, timeout=3)


def own_group() -> dict:
    """Popen arguments for a child that a Ctrl-C at our terminal does not reach."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}
