"""Parts of the installer that need no installation (tests/test_install.py has the whole)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tarfile
import zipfile

import httpx
import pytest

from hearth import install, service
from hearth.hardware import Gpu, Hardware
from hearth.provision import Problem

MANIFEST = {"schema": 1, "version": "1.2.3", "python": "3.12", "files": {"mantel-wheel": {"name": "mantel-1.2.3.whl", "sha256": "ab"}},
            "ollama": {"version": "9", "archives": {"linux-x86_64": [{"url": "a"}], "linux-x86_64-amd": [{"url": "rocm"}]}}}


def _github(monkeypatch, body=None):
    """As GitHub does it: every file is redirected to another host."""
    raw = json.dumps(MANIFEST if body is None else body).encode()

    def get(url, **kw):
        def handle(request):
            if request.url.host == "github.com":
                return httpx.Response(302, headers={"location": "https://release-assets.example/id/uuid?sig=x"})
            return httpx.Response(200, content=raw)

        with httpx.Client(transport=httpx.MockTransport(handle), **kw) as c:
            return c.get(url)

    monkeypatch.setattr(install.httpx, "get", get)
    return raw


def test_a_releases_files_are_beside_its_manifest_wherever_that_was_served_from(monkeypatch):
    raw = _github(monkeypatch)
    tag = "https://github.com/yumima/hearth/releases/download/v1.2.3/"
    for url in (tag + "manifest.json", "https://github.com/yumima/hearth/releases/latest/download/manifest.json", ""):
        rel = install.release(url)
        assert rel.file("mantel-wheel") == (tag + "mantel-1.2.3.whl", "ab"), url
    assert install.release("", hashlib.sha256(raw).hexdigest()).version == "1.2.3"


def test_a_release_that_is_not_to_be_used(monkeypatch):
    _github(monkeypatch)
    with pytest.raises(Problem, match="not the one"):
        install.release("", "0" * 64)
    with pytest.raises(Problem, match="HTTPS"):
        install.release("http://github.com/yumima/hearth/releases/download/v1/manifest.json")
    _github(monkeypatch, {**MANIFEST, "schema": 2})
    with pytest.raises(Problem, match="newer installer"):
        install.release("")


def test_the_engine_for_each_computer():
    rel = install.Release(MANIFEST, "")
    hw = Hardware("linux", "x86_64")
    assert rel.ollama(hw) == ("9", [{"url": "a"}])
    hw.gpus = [Gpu("amd", "x", 16384, None, None, "rocm")]
    assert rel.ollama(hw) == ("9", [{"url": "a"}, {"url": "rocm"}])
    with pytest.raises(Problem, match="no hearth for linux-arm64"):
        rel.ollama(Hardware("linux", "arm64"))


def test_versions_are_compared_as_numbers():
    v = install.version_tuple
    assert v("0.10.0") > v("0.9.9") and v("1.0") > v("0.99.1") and v("x") == ()


@pytest.mark.parametrize("name", ["../evil", "/abs/evil", "a/../../evil", "C:/evil", "..\\evil"])
def test_an_archive_cannot_write_outside_its_folder(tmp_path, name):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(name, "x")
    with pytest.raises(Problem):
        install.unpack(z, tmp_path / "out")
    if ":" not in name and "\\" not in name:
        t = tmp_path / "a.tgz"
        (tmp_path / "f").write_text("x")
        with tarfile.open(t, "w:gz") as f:
            f.add(tmp_path / "f", name)
        try:
            install.unpack(t, tmp_path / "out")
        except tarfile.TarError:
            pass
    assert not (tmp_path / "evil").exists() and not (tmp_path.parent / "evil").exists()


def test_a_zst_archive(tmp_path):
    import zstandard

    plain = tmp_path / "a.tar"
    (tmp_path / "ollama").write_text("x")
    with tarfile.open(plain, "w") as f:
        f.add(tmp_path / "ollama", "bin/ollama")
    z = tmp_path / "a.tar.zst"
    z.write_bytes(zstandard.compress(plain.read_bytes()))
    install.unpack(z, tmp_path / "out")
    assert (tmp_path / "out" / "bin" / "ollama").read_text() == "x"


def test_settings_and_chats_are_put_aside_and_back(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "x"))
    (tmp_path / "h").mkdir()
    (tmp_path / "x" / "mantel").mkdir(parents=True)
    conf, db = tmp_path / "h" / "config.yaml", tmp_path / "x" / "mantel" / "chats.db"
    conf.write_text("old")
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (x)")
    con.execute("INSERT INTO t VALUES ('before')")
    con.commit()  # in the -wal file, with the database open: as mantel leaves it
    saved = install.backup()
    assert install.backup() != saved  # two in one second are two
    con.execute("INSERT INTO t VALUES ('after')")
    con.execute("CREATE TABLE newer (y)")
    con.commit()
    con.close()
    conf.write_text("new")
    install.restore(saved)
    assert conf.read_text() == "old"
    con = sqlite3.connect(db)
    assert con.execute("SELECT x FROM t").fetchall() == [("before",)]
    assert con.execute("SELECT name FROM sqlite_master WHERE name='newer'").fetchall() == []
    con.close()


def test_a_folder_without_the_marker_is_never_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "d"))
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "versions").mkdir()
    with pytest.raises(Problem, match="nothing was removed"):
        install.uninstall(False, False, print)
    assert (tmp_path / "d" / "versions").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="systemd")
def test_a_unit_somebody_wrote_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    ran = []
    monkeypatch.setattr(service, "systemctl", lambda *a: ran.append(a) or 0)
    u = service.unit_path()
    u.parent.mkdir(parents=True)
    u.write_text("[Service]\nExecStart=/mine\n")
    assert service.foreign() == str(u)
    with pytest.raises(service.Foreign):
        service.enable("/data/bin/hearth")
    service.disable(remove=True)
    assert u.read_text() == "[Service]\nExecStart=/mine\n" and ("disable", service.UNIT_NAME) not in ran
    u.unlink()
    service.enable("/a folder/100% hearth")
    assert 'ExecStart="/a folder/100%% hearth" start' in u.read_text() and service.foreign() == ""
    service.disable(remove=True)
    assert not u.exists()


def test_whatever_fails_the_old_version_is_tried_for(monkeypatch, tmp_path):
    """Every step of a switch failing in turn: the old one is started again."""
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path / "h"))
    (tmp_path / "d").mkdir()
    old, new = {"version": "1", "ollama": "a", "mantel": True}, {"version": "2", "ollama": "b", "mantel": True}

    def no(*a, **k):
        raise OSError(28, "No space left on device")

    for broken in ("_stop_everything", "backup", "_set_current", "link_programs"):
        started = []
        with monkeypatch.context() as m:
            m.setattr(install, "_stop_everything", lambda: None)
            m.setattr(install, "link_programs", lambda *a: None)
            m.setattr(install, "_start", lambda v: started.append(v) or (True, "running"))
            install._set_current(old)
            m.setattr(install, broken, no)
            with pytest.raises(Problem, match="did not work. 1 is"):
                install._switch(new, old, True, lambda t: None)
            assert started == ["1"], broken
            assert install.current()["version"] == "1" or broken == "_set_current"
    # one that does not start, and one that was not running before
    for run in (True, False):
        started = []
        with monkeypatch.context() as m:
            m.setattr(install, "_stop_everything", lambda: None)
            m.setattr(install, "link_programs", lambda *a: None)
            m.setattr(install, "_start", lambda v: started.append(v) or (v == "1", "no answer"))
            install._set_current(old)
            with pytest.raises(Problem, match="running again" if run else "is back"):
                install._switch(new, old, run, lambda t: None)
            assert started == (["2", "1"] if run else ["2"]) and install.current() == old


@pytest.mark.skipif(sys.platform == "win32", reason="signals")
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP", "SIGINT"])
def test_asked_to_end_in_the_middle_it_goes_back(monkeypatch, tmp_path, name):
    """finterm closed during Get hearth, or the terminal it ran in."""
    import os
    import signal

    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path / "h"))
    (tmp_path / "d").mkdir()
    old, started = {"version": "1", "ollama": "a", "mantel": True}, []
    monkeypatch.setattr(install, "_stop_everything", lambda: None)
    monkeypatch.setattr(install, "link_programs", lambda *a: None)

    def start(v):
        started.append(v)
        os.kill(os.getpid(), getattr(signal, name))  # (again while it goes back, as at logout)
        return True, "running"

    monkeypatch.setattr(install, "_start", start)
    before = {n: signal.getsignal(getattr(signal, n)) for n in ("SIGTERM", "SIGHUP", "SIGINT")}
    install.ended_in_order()
    try:
        install._set_current(old)
        with pytest.raises((SystemExit, KeyboardInterrupt)):
            install._switch({**old, "version": "2"}, old, True, lambda t: None)
    finally:
        for n, h in before.items():
            signal.signal(getattr(signal, n), h)
    assert started == ["2", "1"] and install.current() == old


def test_nobody_reading_stops_nothing(monkeypatch):
    class Closed:
        def write(self, t):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    monkeypatch.setattr(sys, "stdout", Closed())
    install.say("still going")


@pytest.mark.skipif(sys.platform == "win32", reason="signals")
def test_nohup_stays():
    import signal

    before = {n: signal.getsignal(n) for n in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)}
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        install.ended_in_order()
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN and signal.getsignal(signal.SIGTERM) != before[signal.SIGTERM]
    finally:
        for n, h in before.items():
            signal.signal(n, h)


def test_one_that_cannot_be_made_again_is_put_back(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "d"))
    env = tmp_path / "d" / "versions" / "1.2.3" / "mantel"
    env.mkdir(parents=True)
    (env / ".complete").write_text("")
    (env / "it").write_text("works")
    rel = install.Release({**MANIFEST, "files": {f"mantel-{k}": {"name": k, "sha256": ""} for k in ("wheel", "requirements")}}, "")
    monkeypatch.setattr(install, "fetch", lambda *a: None)
    monkeypatch.setattr(install, "_uv", lambda: "uv")
    stopped = []

    def run(cmd, what, env_vars=None):
        if "venv" in cmd:
            env.mkdir(parents=True)  # half of it
            raise Problem("no disk")

    monkeypatch.setattr(install, "_run", run)
    with pytest.raises(Problem, match="no disk"):
        install.build_env(rel, "mantel", lambda t: None, lambda: stopped.append(1), again=True)
    assert (env / "it").read_text() == "works" and stopped == [1]
    # killed in the middle: the next one finds the one that works, whatever it then does
    env.rename(env.with_name("mantel.before"))
    env.mkdir()
    with pytest.raises(Problem, match="no disk"):
        install.build_env(rel, "mantel", lambda t: None, lambda: stopped.append(1), again=True)
    assert (env / "it").read_text() == "works" and not env.with_name("mantel.before").exists()
    env.rename(env.with_name("mantel.before"))
    assert install.build_env(rel, "mantel", lambda t: None) == env and (env / "it").exists()
    stopped[:] = [1]
    # what cannot be fetched is not even stopped for
    monkeypatch.setattr(install, "fetch", lambda *a: (_ for _ in ()).throw(Problem("no network")))
    with pytest.raises(Problem, match="no network"):
        install.build_env(rel, "mantel", lambda t: None, lambda: stopped.append(1), again=True)
    assert (env / "it").exists() and stopped == [1]
