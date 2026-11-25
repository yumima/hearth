"""The chat window ("Hearth" in the app menu), from the engine's side: where it
is, and putting it there.

    hearth client            open it, with the engine started
    hearth client install    build it, and add it to the app menu
    hearth client uninstall  take it out of the app menu
    hearth client …          its own commands (setup, status, mcp, …)

It is a program of its own, hearth-client, in an environment of its own: what
happens to a window does not happen to the engine. In an installed hearth it is
built from the release hearth is of; in a source checkout from packages/client
(.venv-client). What the menu runs is `hearth client`.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from . import install, paths
from .provision import Problem

_WIN = sys.platform == "win32"
PROGRAM = "hearth-client"


def checkout() -> Path | None:
    """The source tree this hearth runs from, if it has the client in it."""
    p = paths.find_in_checkout("packages", "client", "pyproject.toml")
    return p.parents[2] if p is not None and not install.managed() else None


def _checkout_program(tree: Path) -> Path:
    return install.env_bin(tree / ".venv-client", PROGRAM)


def program() -> str | None:
    """This checkout's own, or the installed hearth's."""
    tree = checkout()
    beside = Path(os.path.abspath(sys.executable)).with_name(install.exe(PROGRAM))
    for p in (*([_checkout_program(tree)] if tree else []), paths.data_dir() / "bin" / install.exe(PROGRAM), beside):
        if p.is_file():
            return str(p)
    return None


def launcher() -> str:
    """The hearth a click runs. (On Windows the one without a console window.)"""
    name = install.exe("hearthw" if _WIN else "hearth")
    if install.managed():
        return str(install.root() / "bin" / name)
    return str(Path(os.path.abspath(sys.executable)).with_name(name))


def _menu(program: str, action: str) -> int:
    return subprocess.run([program, action], env={**os.environ, "HEARTH_PROGRAM": launcher()}, check=False).returncode


def add(rebuild: bool, say: install.Say) -> None:
    """`hearth client install`"""
    tree = checkout()
    if tree is not None:
        _build_from(tree, rebuild, say)  # (nothing to do, if it is as it should be)
        found = str(_checkout_program(tree))
    elif install.managed():
        install.add_client(rebuild, say)
        found = str(install.root() / "bin" / install.exe(PROGRAM))
    else:
        found = program() or ""
        if not found:
            raise Problem("The chat window is not here. Get it with hearth's installer.")
    if _menu(found, "install") != 0:
        raise Problem(f"The chat window is at {found}, and could not be added to the app menu.")


def remove(say: install.Say) -> None:
    """`hearth client uninstall`"""
    found = program()
    if found is None:
        raise Problem("The chat window is not here.")
    _menu(found, "uninstall")


def _works(env: Path) -> bool:
    try:
        return subprocess.run([str(install.env_bin(env, "python")), "-c", "import hearth_client.app"],
                              capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _build_from(tree: Path, again: bool, say: install.Say) -> None:
    env = tree / ".venv-client"
    say(f"Building the chat window in {env}…")
    if uv := shutil.which("uv"):  # (--inexact: the tests' packages of `make build` stay)
        install._run([uv, "sync", "--quiet", "--frozen", "--inexact", *(["--reinstall"] if again else []),
                      "--project", str(tree), "--package", "hearth-client", "--extra", "gui"],
                     "Building the chat window", {"UV_PROJECT_ENVIRONMENT": str(env), "VIRTUAL_ENV": ""})
    elif again or not _works(env):
        # Without uv the versions are not the locked ones; pip takes what is newest in their ranges.
        install._run([sys.executable, "-m", "venv", str(env)], "Making the chat window's environment")
        install._run([str(install.env_bin(env, "python")), "-m", "pip", "install", "--quiet",
                      *(["--force-reinstall"] if again else []), "--upgrade", "--upgrade-strategy", "eager",
                      "-e", f"{tree / 'packages' / 'client'}[gui]"], "Building the chat window")


def open_window(start_engine, args: list[str]) -> int:
    """The window, and beside it the engine: a click shows something at once, and
    the window finds the engine when it is up."""
    import threading

    found = program()
    if found is None:
        print("The chat window is not here — `hearth client install` builds it; plain `hearth` is in the terminal",
              file=sys.stderr)
        return 1
    if start_engine is not None:
        threading.Thread(target=start_engine, daemon=True).start()
    quiet = {"creationflags": subprocess.CREATE_NO_WINDOW} if _WIN and sys.stdout is None else {}
    p = subprocess.Popen([found, *args], env={**os.environ, "HEARTH_PROGRAM": launcher()}, **quiet)
    # A Ctrl-C at the terminal is the program's, which ends in order; what ends us, ends it.
    before = {}
    try:
        before[signal.SIGINT] = signal.signal(signal.SIGINT, signal.SIG_IGN)
        before[signal.SIGTERM] = signal.signal(signal.SIGTERM, lambda *_: p.terminate())
    except ValueError:
        pass  # not the main thread
    try:
        return p.wait()
    finally:
        for n, h in before.items():
            signal.signal(n, h)
