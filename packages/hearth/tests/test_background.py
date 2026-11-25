"""`hearth start`: real processes, no Ollama, no model."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
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


def _answer_once_per_connection(s):
    """Some other program that speaks HTTP."""
    try:
        while True:
            c, _ = s.accept()
            c.recv(1000)
            c.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            c.close()
    except OSError:
        pass


def test_it_serves_and_outlives_the_command(home, tmp_path):
    r = subprocess.run([*EXEC, "start", "--no-manage"],
                       capture_output=True, text=True, timeout=90)
    assert r.returncode == 0, r.stderr
    assert "running (hearth/" in r.stdout
    time.sleep(0.5)  # the command is gone; the gateway is not
    assert httpx.get(f"http://127.0.0.1:{home.bind_port}/admin/version").json()["name"] == "hearth"
    pid = int((tmp_path / "hearth.pid").read_text())
    if hasattr(os, "getsid"):
        assert os.getsid(pid) != os.getsid(0)  # not ended with this terminal
    # a second one finds it, and starts nothing
    r = subprocess.run([*EXEC, "start", "--no-manage"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "already running" in r.stdout
    assert int((tmp_path / "hearth.pid").read_text()) == pid


def test_why_it_could_not_start_is_said(home):
    with socket.socket() as s:  # somebody else's port
        s.bind(("127.0.0.1", home.bind_port))
        s.listen()
        threading.Thread(target=_answer_once_per_connection, args=(s,), daemon=True).start()
        ok, message = background.start(home, [*EXEC, "serve", "--no-manage"], wait_s=30)
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
    monkeypatch.delenv("HEARTH_HOME")
    monkeypatch.setattr(background, "gateway_identity", lambda cfg: None)  # not this machine's own
    monkeypatch.setattr(background, "spawn", lambda argv: pytest.fail(f"spawned {argv}"))
    assert cli.main(["start"]) == 1  # the service failed, and that is said
    assert ran == [("start", cli._UNIT_NAME)]
    # an address of its own is not the service's
    monkeypatch.setattr(background, "spawn", lambda argv: ran.append(argv) or (_ for _ in ()).throw(OSError("x")))
    cli.main(["start", "--bind", f"127.0.0.1:{_free_port()}"])
    assert ran[-1][-2] == "--bind"


def test_starting_changes_nothing_that_is_installed(home, monkeypatch):
    link = Path.home() / ".local" / "bin" / "hearth"
    link.parent.mkdir(parents=True)
    link.symlink_to("/opt/elsewhere/hearth")
    seen = []
    monkeypatch.setattr(background, "start", lambda cfg, argv, service: seen.append(argv) or (True, "running"))
    assert cli.main(["start"]) == 0
    assert os.readlink(link) == "/opt/elsewhere/hearth"
    assert seen == [[sys.executable, "-P", "-m", "hearth", "serve"]]


def test_what_was_asked_for_is_passed_on(home, monkeypatch):
    seen = {}

    def start(cfg, argv, service):
        seen.update(argv=argv, service=service)
        return True, "running"

    monkeypatch.setattr(background, "start", start)
    assert cli.main(["start", "--no-manage", "--log-level", "debug"]) == 0
    assert seen["argv"][5:] == ["--no-manage", "--log-level", "debug"]
    assert seen["service"] is None


def test_an_exposed_gateway_is_not_started_in_the_background(home, monkeypatch):
    monkeypatch.setattr(background, "start", lambda *a: pytest.fail("started"))
    assert cli.main(["start", "--bind", "0.0.0.0:1"]) == 2
    for extra in (["--unsafe-bind", "--api-key", "k"], ["--api-key", "k"]):
        with pytest.raises(SystemExit):
            cli.main(["start", *extra])  # that is `serve`'s


def test_another_config_is_not_the_service(home, tmp_path, monkeypatch):
    unit = tmp_path / "xdg" / "systemd" / "user" / cli._UNIT_NAME
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\n")
    seen = []
    monkeypatch.setattr(background, "start", lambda cfg, argv, service: seen.append(service) or (True, ""))
    cli.main(["start"])  # HEARTH_HOME is set (the fixture's)
    assert seen == [None]


def test_a_service_that_ends_is_not_waited_for(home):
    t = time.monotonic()
    ok, message = background.start(home, [], background.Service(lambda: 0, lambda: False), wait_s=30)
    assert not ok and "service ended" in message
    assert time.monotonic() - t < 10


def test_one_that_is_already_coming_up_is_waited_for(home):
    """Two apps at login: the second one's gateway finds the port taken and ends;
    the first one's answers a moment later."""
    import http.server
    import socketserver

    class Hearth(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Server", "hearth/9")
            self.end_headers()

        def send_response(self, code, message=None):
            self.send_response_only(code, message)

        def log_message(self, *_):
            pass

    class Server(http.server.HTTPServer):
        def server_bind(self):  # HTTPServer's own looks its name up, for half a minute on some machines
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address

    server = Server(("127.0.0.1", home.bind_port), Hearth)  # listening, and silent
    threading.Timer(4, server.serve_forever).start()
    try:
        ok, message = background.start(home, [sys.executable, "-c", "raise SystemExit(1)"], wait_s=30)
        assert ok and "hearth/9" in message
    finally:
        server.shutdown()
        server.server_close()


def test_relative_paths_still_name_the_same_files(home, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEARTH_HOME", "./h")
    elsewhere = str(tmp_path / "abs")
    monkeypatch.setenv("PYTHONPATH", "src" + os.pathsep + elsewhere)
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "queue.py").write_text("raise RuntimeError('a stray file was imported')")
    proc, _ = background.spawn([sys.executable, "-P", "-c",
                                "import os, queue; print(os.environ['HEARTH_HOME'], os.environ['PYTHONPATH'])"])
    assert proc.wait(20) == 0
    assert background.log_path().read_text().split() == [
        str(tmp_path / "h"), str(tmp_path / "src") + os.pathsep + elsewhere]


def test_stop_ends_it_in_order(home, tmp_path):
    """Asked over HTTP, as Windows has to be; its atexit (Ollama) runs."""
    assert background.start(home, [*EXEC, "serve", "--no-manage"])[0]
    pid = int((tmp_path / "hearth.pid").read_text())
    r = subprocess.run([*EXEC, "stop"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert not cli.procs.alive(pid)
    assert "Finished server process" in background.log_path().read_text()  # in order, not killed
    r = subprocess.run([*EXEC, "stop"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "no running hearth gateway" in r.stderr


def test_stop_does_not_end_a_process_that_only_has_its_pid(home, tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "hearth.pid").write_text(str(other.pid))
        # (its command line must not name hearth; tmp_path does not)
        r = subprocess.run([*EXEC, "stop"], capture_output=True, text=True, timeout=60)
        assert r.returncode == 1 and "stale pidfile" in r.stderr
        assert other.poll() is None
    finally:
        other.kill()


def test_shutdown_of_an_app_nobody_serves():
    from fastapi.testclient import TestClient

    from hearth.app import create_app

    c = TestClient(create_app(cfgmod.default_config(), backends_override={}), base_url="http://localhost")
    assert c.post("/admin/shutdown").status_code == 409
    assert c.post("/admin/shutdown", headers={"origin": "https://evil.example"}).status_code == 403


def test_an_exposed_gateway_is_known_by_its_answer(home, monkeypatch):
    """It listens on 0.0.0.0 and wants a key we do not have: that is a hearth, too."""
    import threading

    import uvicorn

    from hearth.app import create_app

    home.bind_host = "0.0.0.0"
    server = uvicorn.Server(uvicorn.Config(create_app(home, allow_any_host=True, api_key="k"), host="127.0.0.1",
                                           port=home.bind_port, log_level="error", server_header=False))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 20
        while background.gateway_identity(home) is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert background.gateway_identity(home).startswith("hearth/")
    finally:
        server.should_exit = True
        t.join(10)
