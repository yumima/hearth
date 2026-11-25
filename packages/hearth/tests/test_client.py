"""`hearth client …`: building the chat window, its menu entry, and opening it."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from hearth import cli, client, install

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="a shell script stands in for the window")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A checkout, with a builder that makes a window which says how it was called."""
    t = tmp_path / "hearth"
    (t / "packages" / "client").mkdir(parents=True)
    (t / "packages" / "client" / "pyproject.toml").write_text("")
    program = t / ".venv-client" / "bin" / "hearth-client"
    built = []

    def run(cmd, what, env=None):
        built.append((cmd, env))
        program.parent.mkdir(parents=True, exist_ok=True)
        program.write_text(f'#!/bin/sh\necho "$* | $HEARTH_PROGRAM" >> "{tmp_path}/calls"\n')
        program.chmod(program.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setattr(client, "checkout", lambda: t)
    monkeypatch.setattr(install, "_run", run)
    monkeypatch.setattr(install, "ended_in_order", lambda: None)  # the test run's own signals
    monkeypatch.setattr(client.shutil, "which", lambda n: "/usr/bin/uv" if n == "uv" else None)
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "data"))
    return type("T", (), {"root": t, "built": built, "program": program,
                          "calls": lambda self: (tmp_path / "calls").read_text().split("\n")[:-1]})()


def test_it_is_built_and_added_to_the_menu(tree):
    assert cli.main(["client", "install"]) == 0
    cmd, env = tree.built[0]
    assert cmd[:3] == ["/usr/bin/uv", "sync", "--quiet"] and "--frozen" in cmd and "--inexact" in cmd
    assert cmd[-4:] == ["--package", "hearth-client", "--extra", "gui"]
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tree.root / ".venv-client")  # not hearth's own
    # what the menu runs is this hearth: it starts the engine, too
    assert tree.calls() == [f"install | {Path(sys.executable).with_name('hearth')}"]
    assert not (Path.home() / ".local" / "bin").exists()  # and no second command
    # again: what is missing or old is put right, whatever had gone wrong; nothing, if nothing had
    assert cli.main(["client", "install"]) == 0 and len(tree.built) == 2 and "--reinstall" not in tree.built[1][0]
    assert cli.main(["client", "install", "--build"]) == 0 and "--reinstall" in tree.built[2][0]


def test_a_checkout_opens_its_own_although_one_is_installed(tree, tmp_path):
    installed = tmp_path / "data" / "bin" / "hearth-client"
    installed.parent.mkdir(parents=True)
    installed.write_text("")
    assert client.program() == str(installed)
    cli.main(["client", "install"])
    assert client.program() == str(tree.program)


def test_without_uv_it_is_built_with_pip(tree, monkeypatch):
    monkeypatch.setattr(client.shutil, "which", lambda n: None)
    assert cli.main(["client", "install"]) == 0
    assert tree.built[0][0][1:3] == ["-m", "venv"] and tree.built[1][0][-1] == f"{tree.root / 'packages' / 'client'}[gui]"
    # one that works is not asked of the network again
    py = tree.root / ".venv-client" / "bin" / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    n = len(tree.built)
    assert cli.main(["client", "install"]) == 0 and len(tree.built) == n
    py.write_text("#!/bin/sh\nexit 1\n")  # cut short
    assert cli.main(["client", "install"]) == 0 and len(tree.built) == n + 2


def test_taking_it_out(tree):
    assert cli.main(["client", "uninstall"]) == 1  # not here
    cli.main(["client", "install"])
    assert cli.main(["client", "uninstall"]) == 0
    assert tree.calls()[-1].startswith("uninstall ") and tree.program.exists()


def test_a_build_that_fails_is_said(tree, monkeypatch, capsys):
    def fail(*a, **k):
        raise client.Problem("Building the chat window failed:\nno network")

    monkeypatch.setattr(install, "_run", fail)
    assert cli.main(["client", "install"]) == 1 and "no network" in capsys.readouterr().err


def test_opening_it_starts_the_engine_its_other_commands_do_not(tree, monkeypatch, capsys):
    import threading

    started = threading.Event()
    monkeypatch.setattr(cli, "cmd_start", lambda a: started.set() if a.background else None)
    assert cli.main(["client"]) == 1 and "hearth client install" in capsys.readouterr().err
    assert not started.is_set()  # nothing to open
    cli.main(["client", "install"])
    for argv in (["client"], ["client", "open"]):
        started.clear()
        assert cli.main(argv) == 0 and started.wait(5), argv
    started.clear()
    for argv in (["client", "status"], ["client", "mcp", "enable", "web"], ["client", "logs", "-f"], ["client", "--help"]):
        assert cli.main(argv) == 0
        assert tree.calls()[-1].split(" | ")[0] == " ".join(argv[1:])  # passed on, unread
    assert not started.is_set()


def test_the_window_does_not_wait_for_the_engine(tree, monkeypatch):
    """A click shows something at once; an engine can take most of a minute."""
    import threading
    import time

    cli.main(["client", "install"])
    end = threading.Event()
    monkeypatch.setattr(cli, "cmd_start", lambda a: end.wait(30))
    t0 = time.monotonic()
    assert cli.main(["client"]) == 0 and time.monotonic() - t0 < 5
    end.set()


def test_how_the_window_ended_is_passed_on(tree):
    cli.main(["client", "install"])
    tree.program.write_text("#!/bin/sh\nexit 7\n")
    assert cli.main(["client", "status"]) == 7


def test_the_checkout_is_found_from_this_file():
    """(Not in an installed copy: there the release is what it is built from.)"""
    t = client.checkout()
    assert t is None or (t / "packages" / "client" / "pyproject.toml").is_file()


def test_stopping_the_engine_is_not_stopping_the_window():
    """…nor the other way round: each knows its own process from the other's."""
    from hearth import procs

    for window in (["/v/bin/python", "/v/bin/hearth-client", "serve"], ["python", "-m", "hearth_client", "serve"],
                   ["/v/bin/python", "/v/bin/hearth", "client"], ["/v/bin/hearth", "client", "start"]):
        assert not procs.runs(window, "hearth", "start"), window
    assert procs.runs(["/v/bin/python", "/v/bin/hearth", "start"], "hearth", "start")
