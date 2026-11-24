"""Where hearth keeps things.

Two places. `home()` is the user's settings and hearth's small state (config,
logs, pid files): ~/.hearth, or HEARTH_HOME. `data_dir()` is what hearth
installs and downloads (its programs, models, voices), in the folder each system
has for that; the installer puts everything there and uninstalling removes it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("HEARTH_HOME") or "~/.hearth").expanduser()


def data_dir() -> Path:
    """finterm's HearthService::install_dir() is this folder, too: it finds
    hearth's program in bin/ there. An installed hearth is in it, wherever it was
    put, and needs no variable to know."""
    prefix = Path(sys.prefix)
    if prefix.parent.parent.name == "versions" and (prefix.parents[2] / ".hearth-install").is_file():
        return prefix.parents[2]
    if env := os.environ.get("HEARTH_DATA"):
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        xdg = os.environ.get("XDG_DATA_HOME") or ""
        base = xdg if os.path.isabs(xdg) else str(Path.home() / ".local" / "share")
    return Path(base) / "hearth"


def current() -> dict:
    """What the installer made current (install.py); {} without an installation."""
    try:
        d = json.loads((data_dir() / "current.json").read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def voices_dir(engine: str) -> Path:
    """Where `engine`'s ("kokoro", "piper") voices are downloaded to."""
    return data_dir() / "voices" / engine


def voice_dirs(engine: str) -> list[Path]:
    """Where they may be: hearth's folder, then the one earlier versions shared
    with finterm, whose uninstaller could remove them."""
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return [voices_dir(engine), Path(xdg) / "finterm" / engine]


def find_in_checkout(*parts: str) -> Path | None:
    """`parts` under a folder above this file: a source checkout's vendor/."""
    for up in Path(__file__).resolve().parents:
        p = up.joinpath(*parts)
        if p.exists():
            return p
        if (up / ".git").exists():
            break
    return None
