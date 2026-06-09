"""CLI command-surface + desktop-launcher tests.

Locks the surface so the recurring app→client→gui→client renames can't silently
break dispatch or the launcher's open-the-window subcommand again. No window is
launched and no real system files are touched — installs are redirected to a
tmp HOME/XDG dir.
"""

from __future__ import annotations

import pytest

from hearth import desktop
from hearth.cli import build_parser


def _dispatch(argv):
    a = build_parser().parse_args(argv)
    return a.func.__name__, getattr(a, "action", "MISSING")


# ── command dispatch ──────────────────────────────────────────────────────────

def test_bare_hearth_is_terminal_chat():
    assert _dispatch([])[0] == "cmd_chat"


def test_client_subcommands_dispatch():
    assert _dispatch(["client"]) == ("cmd_client", None)            # bare → open
    assert _dispatch(["client", "open"]) == ("cmd_client", "open")
    assert _dispatch(["client", "install"]) == ("cmd_client", "install")
    assert _dispatch(["client", "uninstall"]) == ("cmd_client", "uninstall")


def test_chat_and_service_still_dispatch():
    assert _dispatch(["chat"])[0] == "cmd_chat"
    assert _dispatch(["service", "start"]) == ("cmd_service", "start")


@pytest.mark.parametrize("argv", [
    ["gui"], ["install"], ["uninstall"], ["app"],     # old flat / app names are gone
    ["client", "url"],                                # the `url` action was dropped
    ["service", "install"], ["service", "uninstall"],  # removed from `service`
])
def test_removed_commands_are_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# ── launcher: every OS opens the window via `<hearth> client open` ─────────────

def test_launch_argv_is_client_open():
    assert desktop._launch_argv(["/x/hearth"]) == ["/x/hearth", "client", "open"]
    assert desktop._launch_argv(["/usr/bin/python", "-m", "hearth"]) == \
        ["/usr/bin/python", "-m", "hearth", "client", "open"]


def test_windows_lnk_target_and_arguments():
    # _install_windows sets TargetPath = argv[0], Arguments = the rest.
    argv = desktop._launch_argv(["C:/h/hearth.exe"])
    assert argv[0] == "C:/h/hearth.exe"
    assert " ".join(argv[1:]) == "client open"


def test_linux_desktop_entry_runs_client_open(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert desktop._install_linux(["/opt/hearth/bin/hearth"]) == 0
    entry = (tmp_path / "applications" / "hearth-chat.desktop").read_text()
    assert "Exec=/opt/hearth/bin/hearth client open" in entry
    assert "Icon=" in entry


def test_macos_app_launcher_runs_client_open(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))            # Path.home() → tmp
    assert desktop._install_macos(["/opt/hearth/bin/hearth"]) == 0
    launcher = (tmp_path / "Applications" / "Hearth.app" / "Contents" / "MacOS"
                / "hearth-chat").read_text()
    assert launcher.strip().endswith("exec /opt/hearth/bin/hearth client open")
