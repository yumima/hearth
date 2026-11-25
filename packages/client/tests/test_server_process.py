"""`hearth client serve` / `hearth client stop`: real processes, no backend."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import httpx
import psutil
import pytest

from hearth_client import cli
from hearth_client import config as cfgmod


@pytest.fixture
def served(monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfgmod.save(cfgmod.Config(port=port, providers={"hearth": cfgmod.Provider(base_url="http://127.0.0.1:1/v1")}))
    p = subprocess.Popen([sys.executable, "-m", "hearth_client", "serve"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and not cli._server_up(cfgmod.load()):
        time.sleep(0.2)
    yield p, port
    monkeypatch.undo()  # a test may have made killing impossible
    for q in [*psutil.Process(p.pid).children(recursive=True), p] if p.poll() is None else []:
        q.kill()


def test_stop_ends_it_in_order(served, capsys):
    p, port = served
    # (On Windows a venv's python is a launcher: the server is its child.)
    ours = {p.pid, *(c.pid for c in psutil.Process(p.pid).children(recursive=True))}
    assert cli._read_pid() in ours and cli._read_pid() in ours  # asking does not end it
    assert p.poll() is None
    assert cli.main(["stop"]) == 0
    assert p.wait(10) == 0
    assert b"Finished server process" in p.stdout.read()
    assert not cfgmod.pid_path().exists()
    assert cli.main(["stop"]) == 0 and "not running" in capsys.readouterr().out


def test_which_command_lines_are_a_servers():
    for yes in (["/v/bin/python", "/v/bin/hearth-client", "serve"], ["python", "-m", "hearth_client", "serve"],
                ["C:\\m\\Scripts\\hearth-client.exe", "serve"]):
        assert cli._is_server(yes), yes
    for no in (["/home/u/hearth-client/.venv/bin/python", "-m", "pytest"], ["vim", "hearth-client", "serve"],
               ["python", "-c", "hearth client serve"], ["/v/bin/hearth-client"],
               ["/v/bin/python", "/v/bin/hearth", "serve"], ["python", "-m", "hearth", "serve"], []):
        assert not cli._is_server(no), no


def test_another_page_cannot_stop_it(served):
    p, port = served
    r = httpx.post(f"http://127.0.0.1:{port}/api/shutdown", headers={"origin": "https://evil.example"})
    assert r.status_code == 403 and p.poll() is None


def test_a_pid_of_another_program_is_not_ours():
    cfgmod.save(cfgmod.Config(port=1))  # not the port of a client this machine may be running
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        cfgmod.pid_path().parent.mkdir(parents=True, exist_ok=True)
        cfgmod.pid_path().write_text(str(other.pid))
        assert cli._read_pid() is None
        assert cli.main(["stop"]) == 0
        assert other.poll() is None
    finally:
        other.kill()
    cfgmod.pid_path().write_text("not a pid")
    assert cli._read_pid() is None


def test_one_that_cannot_be_stopped_is_not_called_stopped(served, monkeypatch, capsys):
    p, port = served

    def refused(*a, **k):
        raise psutil.AccessDenied(p.pid)

    monkeypatch.setattr(cli.httpx, "post", lambda *a, **k: None)
    monkeypatch.setattr(cli.psutil.Process, "terminate", refused)
    monkeypatch.setattr(cli.psutil.Process, "kill", refused)
    monkeypatch.setattr(cli.psutil.Process, "wait", refused)
    assert cli.main(["stop"]) == 1
    assert "could not stop" in capsys.readouterr().err and cfgmod.pid_path().exists()


def test_a_server_that_cannot_start_says_so(monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        cfgmod.save(cfgmod.Config(port=s.getsockname()[1]))
        r = subprocess.run([sys.executable, "-m", "hearth_client", "serve"], capture_output=True, timeout=60)
    assert r.returncode != 0
