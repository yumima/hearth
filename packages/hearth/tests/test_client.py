"""`hearth client install`: building mantel, and the link and menu entry for it."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from hearth import cli, client, install

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="links; on Windows the installer brings mantel")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A checkout, with a builder that makes a mantel which says how it was called."""
    t = tmp_path / "hearth"
    (t / "packages" / "mantel").mkdir(parents=True)
    (t / "packages" / "mantel" / "pyproject.toml").write_text("")
    built = []

    def run(cmd, what, env=None):
        built.append((cmd, env))
        p = t / ".venv-mantel" / "bin" / "mantel"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'#!/bin/sh\necho "$1 $MANTEL_PROGRAM" >> "{tmp_path}/calls"\n')
        p.chmod(p.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setattr(client, "checkout", lambda: t)
    monkeypatch.setattr(install, "_run", run)
    monkeypatch.setattr(install, "ended_in_order", lambda: None)  # the test run's own signals
    monkeypatch.setattr(client.shutil, "which", lambda n: "/usr/bin/uv" if n == "uv" else None)
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "data"))
    return type("T", (), {"root": t, "built": built, "calls": lambda self: (tmp_path / "calls").read_text().split("\n")[:-1],
                          "link": Path.home() / ".local" / "bin" / "mantel", "program": t / ".venv-mantel" / "bin" / "mantel"})()


def test_it_is_built_linked_and_added_to_the_menu(tree):
    assert cli.main(["client", "install"]) == 0
    cmd, env = tree.built[0]
    assert cmd[:3] == ["/usr/bin/uv", "sync", "--quiet"] and "--frozen" in cmd and "--inexact" in cmd and cmd[-4:] == ["--package", "mantel", "--extra", "gui"]
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tree.root / ".venv-mantel")  # not hearth's own
    assert os.readlink(tree.link) == str(tree.program)
    assert tree.calls() == [f"install {tree.link}"]  # the menu entry is for the name that stays
    # again: nothing is built, and it is in the menu once more
    assert cli.main(["client", "install"]) == 0 and len(tree.built) == 1
    assert cli.main(["client", "install", "--build"]) == 0 and len(tree.built) == 2 and "--reinstall" in tree.built[1][0]
    # one that was interrupted, and another lock file: built once more
    (tree.root / ".venv-mantel" / ".complete").unlink()
    assert cli.main(["client", "install"]) == 0 and len(tree.built) == 3
    (tree.root / "uv.lock").write_text("newer")
    assert cli.main(["client", "install"]) == 0 and len(tree.built) == 4


def test_a_checkout_opens_its_own_although_one_is_installed(tree, tmp_path):
    installed = tmp_path / "data" / "bin" / "mantel"
    installed.parent.mkdir(parents=True)
    installed.write_text("")
    assert client.program() == str(installed)
    cli.main(["client", "install"])
    assert client.program() == str(tree.program)
    (tree.root / ".venv-mantel" / ".complete").unlink()  # its build was cut short
    assert client.program() == str(installed)


def test_a_rebuild_that_is_cut_short_is_not_a_build(tree, monkeypatch):
    cli.main(["client", "install"])

    def cut(*a, **k):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(install, "_run", cut)
        with pytest.raises(KeyboardInterrupt):
            client.add(True, False, print)
    assert cli.main(["client", "install"]) == 0 and len(tree.built) == 2


def test_without_uv_it_is_built_with_pip(tree, monkeypatch):
    monkeypatch.setattr(client.shutil, "which", lambda n: None)
    assert cli.main(["client", "install"]) == 0
    assert tree.built[0][0][1:3] == ["-m", "venv"] and tree.built[1][0][-1] == f"{tree.root / 'packages' / 'mantel'}[gui]"


def test_another_mantel_is_left_unless_asked(tree, tmp_path, capsys):
    other = tmp_path / "old" / "mantel"
    other.parent.mkdir()
    other.write_text("")
    tree.link.parent.mkdir(parents=True)
    tree.link.symlink_to(other)
    assert cli.main(["client", "install"]) == 0
    assert os.readlink(tree.link) == str(other) and "--replace" in capsys.readouterr().out
    assert tree.calls() == [f"install {tree.program}"]  # the menu entry is for this one, by its own path
    assert cli.main(["client", "install", "--replace"]) == 0
    assert os.readlink(tree.link) == str(tree.program) and other.exists()


def test_a_link_that_leads_nowhere_is_replaced(tree):
    """After the old repository was deleted."""
    tree.link.parent.mkdir(parents=True)
    tree.link.symlink_to("/gone/mantel/.venv/bin/mantel")
    assert cli.main(["client", "install"]) == 0 and os.readlink(tree.link) == str(tree.program)


def test_a_file_somebody_put_there_is_never_replaced(tree):
    tree.link.parent.mkdir(parents=True)
    tree.link.write_text("mine")
    assert cli.main(["client", "install", "--replace"]) == 0
    assert tree.link.read_text() == "mine"


def test_taking_it_out(tree, capsys):
    assert cli.main(["client", "uninstall"]) == 1  # not here
    cli.main(["client", "install"])
    assert cli.main(["client", "uninstall"]) == 0
    assert tree.calls()[-1].startswith("uninstall ") and not tree.link.is_symlink() and tree.program.exists()
    tree.link.symlink_to("/somebody/elses")
    cli.main(["client", "uninstall"])
    assert os.readlink(tree.link) == "/somebody/elses"


def test_a_build_that_fails_is_said(tree, monkeypatch, capsys):
    def fail(*a, **k):
        raise client.Problem("Building mantel failed:\nno network")

    monkeypatch.setattr(install, "_run", fail)
    assert cli.main(["client", "install"]) == 1 and "no network" in capsys.readouterr().err
    assert not tree.link.is_symlink()


def test_the_checkout_is_found_from_this_file():
    """(Not in an installed copy: there the release is what mantel is built from.)"""
    t = client.checkout()
    assert t is None or (t / "packages" / "mantel" / "pyproject.toml").is_file()
