"""mantel, the chat window, from hearth's side: where it is, and putting it there.

    hearth client            open it
    hearth client install    build it if it is not here, and add it to the app menu
    hearth client uninstall  take it out of the app menu

In an installed hearth it is built from the release hearth is of; in a source
checkout from packages/mantel, in an environment of its own (.venv-mantel).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import install, paths
from .provision import Problem

_WIN = sys.platform == "win32"


def checkout() -> Path | None:
    """The source tree this hearth runs from, if it has mantel in it."""
    p = paths.find_in_checkout("packages", "mantel", "pyproject.toml")
    return p.parents[2] if p is not None and not install.managed() else None


def _checkout_program(tree: Path) -> Path:
    return install.env_bin(tree / ".venv-mantel", "mantel")


def program() -> str | None:
    """Where the installer puts it, this checkout's own, beside this program, or on PATH."""
    tree = checkout()
    beside = Path(os.path.abspath(sys.executable)).with_name(install.exe("mantel"))
    for p in (paths.data_dir() / "bin" / install.exe("mantel"), *([_checkout_program(tree)] if tree else []), beside):
        if p.is_file():
            return str(p)
    return shutil.which("mantel")


def add(rebuild: bool, replace: bool, say: install.Say) -> None:
    """`hearth client install`"""
    tree = checkout()
    if tree is not None:
        p = _checkout_program(tree)
        if rebuild or not p.is_file():
            _build_from(tree, say)
        shown = _link(p, replace, say)
    elif install.managed():
        p = install.root() / "bin" / install.exe("mantel")
        if rebuild or not p.exists():
            install.add_mantel(rebuild, say)
        shown = p
    else:
        found = program()
        if found is None:
            raise Problem("There is no mantel here to add. Get it with hearth's installer.")
        shown = Path(found)
    r = subprocess.run([str(shown), "install"], env={**os.environ, "MANTEL_PROGRAM": str(shown)}, check=False)
    if r.returncode != 0:
        raise Problem(f"mantel is at {shown}, and could not be added to the app menu.")


def remove(say: install.Say) -> None:
    """`hearth client uninstall`"""
    found = program()
    if found is None:
        raise Problem("mantel is not here.")
    subprocess.run([found, "uninstall"], env={**os.environ, "MANTEL_PROGRAM": found}, check=False)
    tree = checkout()
    link = Path.home() / ".local" / "bin" / "mantel"
    if tree is not None and link.is_symlink() and os.path.abspath(os.readlink(link)) == str(_checkout_program(tree)):
        link.unlink()
        say(f"removed {link}")


def _build_from(tree: Path, say: install.Say) -> None:
    env = tree / ".venv-mantel"
    say(f"Building mantel in {env}…")
    if uv := shutil.which("uv"):
        install._run([uv, "sync", "--quiet", "--frozen", "--project", str(tree), "--package", "mantel"],
                     "Building mantel", {"UV_PROJECT_ENVIRONMENT": str(env), "VIRTUAL_ENV": ""})
        return
    # Without uv the versions are not the locked ones; pip takes what is newest in their ranges.
    install._run([sys.executable, "-m", "venv", str(env)], "Making mantel's environment")
    install._run([str(install.env_bin(env, "python")), "-m", "pip", "install", "--quiet", "-e", str(tree / "packages" / "mantel")],
                 "Building mantel")


def _link(p: Path, replace: bool, say: install.Say) -> Path:
    """~/.local/bin/mantel, where it is free: the name the menu entry and a shell use."""
    if _WIN:
        return p
    link = Path.home() / ".local" / "bin" / "mantel"
    here = os.path.abspath(os.readlink(link)) if link.is_symlink() else ""
    if here == str(p):
        return link
    if link.is_symlink() and (replace or not os.path.exists(link)):  # asked for, or it leads nowhere any more
        link.unlink()
    if link.exists() or link.is_symlink():
        say(f"{link} is another mantel{f' ({here})' if here else ''} and is left as it is; the menu entry is for this one. "
            "To have it replaced: hearth client install --replace")
        return p
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(p)
    say(f"{link} → {p}")
    return link
