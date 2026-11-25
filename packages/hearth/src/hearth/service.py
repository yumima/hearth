"""Starting hearth at login — with what each system has for that, and no more:
a systemd user unit (or an XDG autostart entry), a LaunchAgent, the Run key.

Only what hearth's installer made is changed: a unit somebody wrote themselves is
theirs, and is left as it is.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import paths

UNIT_NAME = "hearth.service"
LABEL = "io.github.yumima.hearth"
MARK = "# made by hearth's installer"
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


class Foreign(Exception):
    """There is one that is not ours."""


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))


def unit_path() -> Path:
    return _config_home() / "systemd" / "user" / UNIT_NAME


def _autostart_path() -> Path:
    return _config_home() / "autostart" / "hearth.desktop"


def _agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _env() -> dict:
    env = dict(os.environ)
    if hasattr(os, "getuid"):
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return env


def systemctl(*args: str) -> int:
    sc = shutil.which("systemctl")
    if not sc:
        return 127
    return subprocess.run([sc, "--user", *args], env=_env(), stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode


def has_systemd() -> bool:
    return sys.platform.startswith("linux") and systemctl("show-environment") == 0


def _ours(p: Path) -> bool:
    try:
        return MARK in p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def _write(p: Path, text: str) -> None:
    if p.exists() and not _ours(p):
        raise Foreign(f"{p} was not made by hearth's installer; it is left as it is")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _unit(program: str) -> str:
    quoted = program.replace("%", "%%").replace('"', '\\"')
    return f"""{MARK}
[Unit]
Description=hearth — local AI engine on 127.0.0.1:11435

[Service]
ExecStart="{quoted}" serve
Restart=on-failure
RestartSec=5
# The Ollama it starts goes with it, in order, and is given the time to.
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=default.target
"""


def enable(program: str) -> None:
    """Start the engine of `program` at login, from the next one on."""
    if sys.platform == "win32":
        import winreg

        if f := foreign():
            raise Foreign(f"{f} was not made by hearth's installer; it is left as it is")
        # hearthw: the same without a console window, which would be there all day.
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as k:
            winreg.SetValueEx(k, "hearth", 0, winreg.REG_SZ, f'"{program}" start')
    elif sys.platform == "darwin":
        p = _agent_path()
        if p.exists() and not _agent_ours(p):
            raise Foreign(f"{p} was not made by hearth's installer; it is left as it is")
        p.parent.mkdir(parents=True, exist_ok=True)
        log = str(paths.home() / "hearth.log")
        p.write_bytes(plistlib.dumps({
            "Label": LABEL, "ProgramArguments": [program, "serve"], "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False}, "StandardOutPath": log, "StandardErrorPath": log,
            "ProcessType": "Interactive", "HearthInstaller": True}))
    elif has_systemd():
        _write(unit_path(), _unit(program))
        systemctl("daemon-reload")
        if systemctl("enable", UNIT_NAME) != 0:
            raise OSError("systemctl could not enable hearth.service")
    else:
        _write(_autostart_path(), f"[Desktop Entry]\n{MARK}\nType=Application\nName=hearth\n"
                                  f"Exec={shlex.quote(program)} start\nX-GNOME-Autostart-enabled=true\nNoDisplay=true\n")


def _agent_ours(p: Path) -> bool:
    try:
        return plistlib.loads(p.read_bytes()).get("HearthInstaller") is True
    except (OSError, ValueError, plistlib.InvalidFileException):
        return False


def disable(remove: bool = False) -> None:
    """Not at login any more. `remove`: nor is there anything left of it."""
    if sys.platform == "win32":
        import winreg

        if not foreign():
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
                    winreg.DeleteValue(k, "hearth")
            except OSError:
                pass
    elif sys.platform == "darwin":
        p = _agent_path()
        if _agent_ours(p):
            p.unlink()
    else:
        u = unit_path()
        if _ours(u):
            systemctl("disable", UNIT_NAME)
            if remove:
                u.unlink()
                systemctl("daemon-reload")
        if _ours(_autostart_path()):
            _autostart_path().unlink()


def enabled() -> bool:
    if sys.platform == "win32":
        return bool(_run_value())
    if sys.platform == "darwin":
        return _agent_path().exists()
    return (unit_path().exists() and systemctl("is-enabled", UNIT_NAME) == 0) or _autostart_path().exists()


def _run_value() -> str:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as k:
            return str(winreg.QueryValueEx(k, "hearth")[0])
    except OSError:
        return ""


def foreign() -> str:
    """The path of one that is not ours, "" if there is none."""
    if sys.platform == "win32":
        value = _run_value()
        ours = str(paths.data_dir() / "bin").lower()
        return f"the login entry {value}" if value and ours not in value.lower() else ""
    candidates = {"win32": [], "darwin": [_agent_path()]}.get(sys.platform, [unit_path(), _autostart_path()])
    for p in candidates:
        if p.exists() and not (_agent_ours(p) if sys.platform == "darwin" else _ours(p)):
            return str(p)
    return ""
