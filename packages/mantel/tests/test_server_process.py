"""`mantel serve` / `mantel stop`: real processes, no backend."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import httpx
import pytest

from mantel import cli
from mantel import config as cfgmod


@pytest.fixture
def served(monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfgmod.save(cfgmod.Config(port=port, providers={"hearth": cfgmod.Provider(base_url="http://127.0.0.1:1/v1")}))
    p = subprocess.Popen([sys.executable, "-m", "mantel", "serve"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and not cli._server_up(cfgmod.load()):
        time.sleep(0.2)
    yield p, port
    if p.poll() is None:
        p.kill()


def test_stop_ends_it_in_order(served, capsys):
    p, port = served
    assert cli._read_pid() == p.pid and cli._read_pid() == p.pid  # asking does not end it
    assert p.poll() is None
    assert cli.main(["stop"]) == 0
    assert p.wait(10) == 0
    assert b"Finished server process" in p.stdout.read()
    assert not cfgmod.pid_path().exists()
    assert cli.main(["stop"]) == 0 and "not running" in capsys.readouterr().out


def test_another_page_cannot_stop_it(served):
    p, port = served
    r = httpx.post(f"http://127.0.0.1:{port}/api/shutdown", headers={"origin": "https://evil.example"})
    assert r.status_code == 403 and p.poll() is None


def test_a_pid_of_another_program_is_not_ours():
    cfgmod.save(cfgmod.Config(port=1))  # not the port of a mantel this machine may be running
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
