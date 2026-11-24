"""mantel, the chat window, from hearth's side: where it is, and putting it there.

    hearth client            open it
    hearth client install    build it if it is not here, and add it to the app menu
    hearth client uninstall  take it out of the app menu

In an installed hearth it is built from the release hearth is of; in a source
checkout from packages/mantel, in an environment of its own (.venv-mantel).
"""

from __future__ import annotations

import hashlib
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
    """This checkout's own, where the installer puts it, beside this program, or on PATH."""
    tree = checkout()
    beside = Path(os.path.abspath(sys.executable)).with_name(install.exe("mantel"))
    for p in (*([_checkout_program(tree)] if tree else []), paths.data_dir() / "bin" / install.exe("mantel"), beside):
        if p.is_file():
            return str(p)
    return shutil.which("mantel")


def add(rebuild: bool, replace: bool, say: install.Say) -> None:
    """`hearth client install`"""
    tree = checkout()
    if tree is not None:
        p = _checkout_program(tree)
        if rebuild or not p.is_file() or _built(tree) != _locked(tree):
            _build_from(tree, rebuild, say)
        shown = _link(p, replace, say)
    elif install.managed():
        p = install.root() / "bin" / install.exe("mantel")
        if rebuild or not (install.root() / "versions" / install.current()["version"] / "mantel" / ".complete").is_file() \
                or not p.exists():
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


def _locked(tree: Path) -> str:
    """What mantel is made of: another lock file is another mantel."""
    p = tree / "uv.lock"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "-"


def _built(tree: Path) -> str:
    """Of the last build that was finished."""
    try:
        return (tree / ".venv-mantel" / ".complete").read_text()
    except OSError:
        return ""


def _build_from(tree: Path, again: bool, say: install.Say) -> None:
    env = tree / ".venv-mantel"
    say(f"Building mantel in {env}…")
    if uv := shutil.which("uv"):  # (--inexact: the tests' packages of `make build` stay)
        install._run([uv, "sync", "--quiet", "--frozen", "--inexact", *(["--reinstall"] if again else []),
                      "--project", str(tree), "--package", "mantel", "--extra", "gui"],
                     "Building mantel", {"UV_PROJECT_ENVIRONMENT": str(env), "VIRTUAL_ENV": ""})
    else:
        # Without uv the versions are not the locked ones; pip takes what is newest in their ranges.
        install._run([sys.executable, "-m", "venv", str(env)], "Making mantel's environment")
        install._run([str(install.env_bin(env, "python")), "-m", "pip", "install", "--quiet",
                      *(["--force-reinstall"] if again else []), "-e", f"{tree / 'packages' / 'mantel'}[gui]"],
                     "Building mantel")
    (env / ".complete").write_text(_locked(tree))


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
