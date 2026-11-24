"""Other processes, and the Ollama hearth keeps running — with a stand-in for
Ollama (a small HTTP server), real processes, and no model."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

from hearth import ollama_supervisor as osup
from hearth import paths, procs
from hearth.ollama_supervisor import Supervisor


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _until(what, wait_s=20.0):
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if what():
            return True
        time.sleep(0.1)
    return False


# ── processes ─────────────────────────────────────────────────────────────────

def test_alive_and_asking_does_not_end_it():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert procs.alive(p.pid) and procs.alive(p.pid)
        assert p.poll() is None
        assert "time.sleep" in procs.command_line(p.pid)
        assert procs.started_at(p.pid) > 0
    finally:
        procs.end_tree(p.pid)
    assert not procs.alive(p.pid) and procs.started_at(p.pid) == 0 and procs.command_line(p.pid) == ""


def test_which_command_lines_are_a_gateways():
    yes = (["/v/bin/python3", "/v/bin/hearth", "start"], ["python", "-m", "hearth", "start", "--bind", "x"],
           ["python", "-P", "-m", "hearth", "start"], ["C:\\h\\Scripts\\hearth.exe", "start"], ["hearth", "start"],
           ["C:\\py\\python.exe", "C:\\h\\Scripts\\hearth.exe", "start"],
           ["/home/hearth/.venv/bin/python", "-m", "hearth", "start"])
    no = (["/home/u/hearth/.venv/bin/python", "-m", "pytest"], ["/v/bin/python", "/v/bin/hearth", "chat"],
          ["python", "-c", "hearth start"], ["vim", "hearth", "start"], ["/src/hearth/run.sh", "start"],
          ["grep", "-r", "hearth", "start"], ["python", "-m", "uvicorn", "hearth.app"], [], ["/usr/bin/hearth"])
    assert [a for a in yes if not procs.runs(a, "hearth", "start")] == []
    assert [a for a in no if procs.runs(a, "hearth", "start")] == []


def test_a_tree_ends_with_its_top():
    code = ("import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
            "print(c.pid, flush=True); time.sleep(60)")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    child = int(p.stdout.readline())
    procs.end_tree(p.pid)
    assert _until(lambda: not procs.alive(child) and not procs.alive(p.pid), 5)
    procs.end_tree(p.pid)  # nothing to do, and no error


def test_who_listens():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        assert procs.listening_on(port) is None  # bound is not listening
        s.listen()
        assert procs.listening_on(port) == os.getpid()


# ── Ollama ────────────────────────────────────────────────────────────────────

FAKE = textwrap.dedent("""
    import http.server, json, os, sys
    print("fake ollama here", sys.argv[1:], flush=True)
    if os.environ.get("FAKE_DIES"):
        sys.exit(3)
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/die":
                os._exit(9)
            body = json.dumps({"version": "fake", **{k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")}}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
        def log_message(self, *_): pass
    class Server(http.server.HTTPServer):
        def server_bind(self):  # HTTPServer's own looks its name up, for half a minute on some machines
            __import__("socketserver").TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address
    host, port = os.environ["OLLAMA_HOST"].split(":")
    Server((host, int(port)), H).serve_forever()
""")


@pytest.fixture
def ollama(tmp_path, monkeypatch):
    """A program called ollama, a home, and a free address."""
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path / "hearth"))
    script = tmp_path / "ollama_fake.py"
    script.write_text(FAKE)
    if sys.platform == "win32":
        exe = tmp_path / "ollama.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        exe = tmp_path / "ollama"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        exe.chmod(0o755)
    monkeypatch.setenv("HEARTH_OLLAMA_BIN", str(exe))
    monkeypatch.setattr(osup, "_LOOK_S", 0.2)
    monkeypatch.setattr(osup, "_BACKOFF_S", (0.1, 0.2))
    for name in [n for n in os.environ if n.startswith("OLLAMA_")]:
        monkeypatch.delenv(name)
    base = f"http://127.0.0.1:{_free_port()}"
    started = []

    def make(models_dir=""):
        started.append(Supervisor(base, models_dir))
        return started[-1]

    yield make
    for s in started:
        s.stop()


def _kill(base):
    try:
        httpx.get(base + "/die", timeout=2)
    except httpx.HTTPError:
        pass


def test_it_is_started_with_our_settings_and_its_own_models(ollama, tmp_path):
    s = ollama(str(tmp_path / "models"))
    assert s.start() and osup.is_up(s.base_url)
    env = httpx.get(s.base_url + "/api/version").json()
    assert env["OLLAMA_MODELS"] == str(tmp_path / "models")
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0" and env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert "fake ollama here ['serve']" in osup.last_lines()  # its output is kept
    s.stop()
    assert not osup.is_up(s.base_url)
    assert not (paths.home() / "ollama.pid").exists()


def test_without_a_folder_of_its_own_ollamas_is_used(ollama):
    s = ollama()
    assert s.start()
    assert "OLLAMA_MODELS" not in httpx.get(s.base_url + "/api/version").json()


def test_it_is_started_again_when_it_dies(ollama):
    s = ollama()
    assert s.start()
    first = s._pid
    for _ in range(2):
        _kill(s.base_url)
        assert _until(lambda: not osup.is_up(s.base_url)) and _until(lambda: osup.is_up(s.base_url))
    assert s._pid != first


def test_one_that_keeps_dying_is_tried_at_a_pace_and_can_be_stopped(ollama, monkeypatch):
    monkeypatch.setenv("FAKE_DIES", "1")
    s = ollama()
    t = time.monotonic()
    assert s.start()  # it is ours to keep trying
    assert time.monotonic() - t < 10  # and nobody waits for one that ended
    assert _until(lambda: s._failures >= 3)
    s.stop()
    n = s._failures
    time.sleep(1)
    assert s._failures == n


def test_somebody_elses_is_used_and_left_alone(ollama):
    theirs = ollama()
    assert theirs.start()
    (paths.home() / "ollama.pid").unlink()  # not known to us
    ours = Supervisor(theirs.base_url)
    assert ours.start() is False and osup.is_up(ours.base_url)
    ours.stop()
    assert osup.is_up(theirs.base_url)


def _owner_is_gone():
    p = paths.home() / "ollama.pid"
    p.write_text(json.dumps({**json.loads(p.read_text()), "owner": 2**22 - 7, "owner_started": 1.0}))


def test_one_left_by_a_killed_hearth_is_taken_over(ollama):
    left = ollama()
    assert left.start(keep_up=False)
    _owner_is_gone()
    ours = Supervisor(left.base_url)
    assert ours.start() and ours._pid == left._pid
    _kill(ours.base_url)
    assert _until(lambda: not osup.is_up(ours.base_url)) and _until(lambda: osup.is_up(ours.base_url))
    ours.stop()
    assert not osup.is_up(ours.base_url)


def test_one_whose_hearth_lives_is_not_taken_over(ollama):
    """A second gateway of the same home (hearth start --bind …)."""
    first = ollama()
    assert first.start()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        p = paths.home() / "ollama.pid"
        p.write_text(json.dumps({**json.loads(p.read_text()), "owner": other.pid,
                                 "owner_started": procs.started_at(other.pid)}))
        second = Supervisor(first.base_url)
        assert second.start() is False
        second.stop()
        assert osup.is_up(first.base_url) and p.exists()
    finally:
        other.kill()


def test_one_at_another_address_is_not_taken_over(ollama):
    first = ollama()
    assert first.start(keep_up=False)
    _owner_is_gone()
    assert Supervisor("http://127.0.0.1:1")._adopt() is False


def test_a_pid_that_is_another_process_by_now_is_not_taken_over(ollama):
    theirs = ollama()
    assert theirs.start()
    _owner_is_gone()
    good = json.loads((paths.home() / "ollama.pid").read_text())
    for left in ({**good, "started": 1.0}, {**good, "pid": os.getpid(), "started": procs.started_at(os.getpid())},
                 {**good, "pid": "x"}, {k: v for k, v in good.items() if k != "owner"}, [], {}):
        (paths.home() / "ollama.pid").write_text(json.dumps(left))
        assert Supervisor(theirs.base_url)._adopt() is False, left
    (paths.home() / "ollama.pid").write_text("not json")
    assert Supervisor(theirs.base_url)._adopt() is False


def test_it_comes_back_after_a_commands_own_one(ollama, monkeypatch):
    """`hearth pull` while the gateway's Ollama is down for a restart."""
    monkeypatch.setattr(osup, "_BACKOFF_S", (2.0,))
    gateway = ollama()
    assert gateway.start()
    _kill(gateway.base_url)
    assert _until(lambda: gateway._failures == 1)
    command = Supervisor(gateway.base_url)
    assert command.start(keep_up=False)
    time.sleep(3)  # past the gateway's wait
    assert gateway._ended()  # the command's is used
    command.stop()
    assert _until(lambda: osup.is_up(gateway.base_url)) and not gateway._ended()


def test_nothing_is_started_once_it_is_stopping(ollama, monkeypatch):
    s = ollama()
    assert s.start()
    asked = []

    def slow_is_up(base_url, timeout=1.5):
        asked.append(1)
        time.sleep(0.6)  # stop() arrives while the watcher is here
        return False

    monkeypatch.setattr(osup, "is_up", slow_is_up)
    _kill(s.base_url)
    assert _until(lambda: asked)
    s.stop()
    time.sleep(1.5)
    with pytest.raises(httpx.HTTPError):
        httpx.get(s.base_url + "/api/version", timeout=1)


def test_what_ends_while_the_gateway_goes_down_stays_ended(ollama):
    s = ollama()
    assert s.start()
    s.stop_watching()
    _kill(s.base_url)
    time.sleep(1.5)
    assert not osup.is_up(s.base_url)


def test_no_program_at_all(ollama, monkeypatch, tmp_path):
    monkeypatch.setattr(osup, "ollama_binary", lambda: None)
    assert ollama().start() is False


def test_where_the_program_is_looked_for(monkeypatch, tmp_path):
    monkeypatch.delenv("HEARTH_OLLAMA_BIN", raising=False)
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path))
    monkeypatch.setattr(osup.paths, "find_in_checkout", lambda *p: None)
    monkeypatch.setattr(osup, "which", lambda n: "/on/path/ollama")
    assert osup.ollama_binary() == "/on/path/ollama"
    exe = tmp_path / "ollama" / ("ollama.exe" if sys.platform == "win32" else "ollama")
    exe.parent.mkdir()
    exe.write_text("")
    assert osup.ollama_binary() == str(exe)  # the installed one comes first
    nested = tmp_path / "ollama" / "bin" / exe.name
    nested.parent.mkdir()
    nested.write_text("")
    assert osup.ollama_binary() == str(nested)


# ── folders ───────────────────────────────────────────────────────────────────

def test_folders(monkeypatch, tmp_path):
    monkeypatch.delenv("HEARTH_DATA", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    assert paths.data_dir() == Path.home() / ".local" / "share" / "hearth"
    monkeypatch.setenv("XDG_DATA_HOME", "relative")  # the specification: ignored
    assert paths.data_dir() == Path.home() / ".local" / "share" / "hearth"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert paths.data_dir() == tmp_path / "hearth"
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert paths.data_dir() == Path.home() / "Library" / "Application Support" / "hearth"
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "L"))
    assert paths.data_dir() == tmp_path / "L" / "hearth"
    monkeypatch.setenv("HEARTH_HOME", "")  # empty is not a folder
    assert paths.home() == Path("~/.hearth").expanduser()


def test_voices_come_from_hearths_folder_then_the_one_it_shared(monkeypatch, tmp_path):
    from hearth import kokoro_tts
    from hearth.routes import v1

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "x"))
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "d"))
    monkeypatch.delenv("HEARTH_KOKORO_DIR", raising=False)
    old, new = tmp_path / "x" / "finterm", tmp_path / "d" / "voices"
    for d, names in ((old / "piper", ("en_US-amy-medium", "it_IT-paola-medium")), (new / "piper", ("en_US-amy-medium",))):
        d.mkdir(parents=True)
        for n in names:
            (d / f"{n}.onnx").write_text("")
    assert v1._find_voice(None) == new / "piper" / "en_US-amy-medium.onnx"
    assert v1._find_voice("it_IT-paola-medium") == old / "piper" / "it_IT-paola-medium.onnx"
    assert v1._find_voice("../../../etc/passwd") == new / "piper" / "en_US-amy-medium.onnx"
    assert kokoro_tts.model_dir() == new / "kokoro"  # nowhere yet: where it would go
    (old / "kokoro").mkdir()
    (old / "kokoro" / kokoro_tts.MODEL_FILE).write_text("")
    assert kokoro_tts.model_dir() == new / "kokoro"  # half of it is none of it
    (old / "kokoro" / kokoro_tts.VOICES_FILE).write_text("")
    assert kokoro_tts.model_dir() == old / "kokoro"


def test_a_gateway_going_down_stops_watching_first():
    from fastapi.testclient import TestClient

    from hearth import config as cfgmod
    from hearth.app import create_app

    app = create_app(cfgmod.default_config(), backends_override={})
    called = []
    app.state.at_shutdown.append(lambda: called.append(1))
    with TestClient(app, base_url="http://localhost"):
        assert not called
    assert called == [1]
