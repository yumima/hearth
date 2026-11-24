"""`hearth start --background`: real processes, no Ollama, no model."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from hearth import background, cli
from hearth import config as cfgmod

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A hearth of its own: config, log and pidfile in tmp_path; no service."""
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    monkeypatch.delenv("HEARTH_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("PYTHONPATH", SRC)
    cfg = cfgmod.default_config()
    cfg.bind_port = _free_port()
    cfg.path = tmp_path / "config.yaml"
    cfgmod.save(cfg)
    yield cfg
    pid = tmp_path / "hearth.pid"
    if pid.exists():
        os.kill(int(pid.read_text()), signal.SIGTERM)


EXEC = [sys.executable, "-m", "hearth"]


def test_it_serves_and_outlives_the_command(home, tmp_path):
    r = subprocess.run([*EXEC, "start", "--background", "--no-manage"],
                       capture_output=True, text=True, timeout=90)
    assert r.returncode == 0, r.stderr
    assert "running (hearth/" in r.stdout
    time.sleep(0.5)  # the command is gone; the gateway is not
    assert httpx.get(f"http://127.0.0.1:{home.bind_port}/admin/version").json()["name"] == "hearth"
    pid = int((tmp_path / "hearth.pid").read_text())
    if hasattr(os, "getsid"):
        assert os.getsid(pid) != os.getsid(0)  # not ended with this terminal
    # a second one finds it, and starts nothing
    r = subprocess.run([*EXEC, "start", "--background", "--no-manage"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "already running" in r.stdout
    assert int((tmp_path / "hearth.pid").read_text()) == pid


def test_why_it_could_not_start_is_said(home):
    with socket.socket() as s:  # somebody else's port
        s.bind(("127.0.0.1", home.bind_port))
        s.listen()
        s.settimeout(0)
        ok, message = background.start(home, EXEC, ["--no-manage"], wait_s=30)
    assert not ok
    assert "ended (exit 1)" in message and "already in use" in message


def test_only_its_own_output_is_quoted(home):
    background.log_path().write_text("an old error\n")
    ok, message = background.start(home, [sys.executable, "-c", "raise SystemExit('new trouble')"])
    assert not ok and "new trouble" in message and "old error" not in message


def test_a_program_that_is_not_there(home):
    ok, message = background.start(home, ["/nonexistent/hearth"])
    assert not ok and "could not run /nonexistent/hearth" in message


def test_one_that_never_answers(home):
    ok, message = background.start(
        home, [sys.executable, "-c", "import time; time.sleep(5)"], wait_s=1)
    assert not ok and "did not answer" in message and "hearth.log" in message


def test_a_large_log_is_moved_aside(home):
    background.log_path().write_bytes(b"x" * (background._LOG_MAX + 1))
    background.start(home, [sys.executable, "-c", "print('fresh')"])
    assert background.log_path().read_text().strip() == "fresh"
    assert background.log_path().with_suffix(".log.1").stat().st_size > background._LOG_MAX


def test_an_installed_service_is_what_starts(home, tmp_path, monkeypatch):
    unit = tmp_path / "xdg" / "systemd" / "user" / cli._UNIT_NAME
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\n")
    ran = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(cli, "_systemctl", lambda *a: ran.append(a) or 1)
    monkeypatch.setattr(background, "spawn", lambda argv: pytest.fail(f"spawned {argv}"))
    assert cli.main(["start", "--background"]) == 1  # the service failed, and that is said
    assert ran == [("start", cli._UNIT_NAME)]
    # an address of its own is not the service's
    monkeypatch.setattr(background, "spawn", lambda argv: ran.append(argv) or (_ for _ in ()).throw(OSError("x")))
    cli.main(["start", "--background", "--bind", f"127.0.0.1:{_free_port()}"])
    assert ran[-1][-2] == "--bind"


def test_starting_changes_nothing_that_is_installed(home, monkeypatch):
    link = Path.home() / ".local" / "bin" / "hearth"
    link.parent.mkdir(parents=True)
    link.symlink_to("/opt/elsewhere/hearth")
    seen = []
    monkeypatch.setattr(background, "start", lambda cfg, argv, *a: seen.append(argv) or (True, "running"))
    assert cli.main(["start", "--background"]) == 0
    assert os.readlink(link) == "/opt/elsewhere/hearth"
    assert seen == [[sys.executable, "-m", "hearth"]]


def test_what_was_asked_for_is_passed_on(home, monkeypatch):
    seen = {}

    def start(cfg, hearth_exec, extra, service):
        seen.update(extra=extra, service=service)
        return True, "running"

    monkeypatch.setattr(background, "start", start)
    assert cli.main(["start", "--background", "--unsafe-bind", "--api-key", "k", "--no-manage",
                     "--log-level", "debug"]) == 0
    assert seen["extra"] == ["--no-manage", "--unsafe-bind", "--api-key", "k", "--log-level", "debug"]
    assert seen["service"] is None
    assert cli.main(["start", "--background", "--unsafe-bind"]) == 2  # still wants its key
