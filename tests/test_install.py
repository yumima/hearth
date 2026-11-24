"""The whole life of an installation, with the real scripts, real uv and real
environments: install, update, an update that does not start, going back,
removing. A folder is served as the release; Ollama is a stand-in.

Run with HEARTH_E2E=1 (CI does): it downloads uv, Python and packages.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

import httpx
import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
WIN = sys.platform == "win32"

pytestmark = [pytest.mark.skipif(not os.environ.get("HEARTH_E2E"), reason="HEARTH_E2E is not set"),
              pytest.mark.timeout(1200)]

FAKE_OLLAMA = '''
import http.server, json, os, socketserver
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"version": "fake", "models": [], "dir": os.environ.get("OLLAMA_MODELS")}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_): pass
def main():
    host, port = os.environ["OLLAMA_HOST"].split(":")
    socketserver.TCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer((host, int(port)), H).serve_forever()
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(cmd, env=None, ok=True, timeout=900, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, timeout=timeout,
                       encoding="utf-8", errors="replace", **kw)
    if ok is not None:
        assert (r.returncode == 0) is ok, f"{cmd}\n{r.stdout}\n{r.stderr}"
    return r


class World:
    """A computer with nothing on it, and a place releases come from."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.home = tmp / "home"
        self.releases = tmp / "releases"
        self.releases.mkdir(parents=True)
        self.home.mkdir()
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(self.releases))
        http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.port, self.ollama_port = _free_port(), _free_port()
        # Nothing of the machine the test runs on: not its home, its session
        # (systemd), its Python or its PATH's hearth.
        keep = ("PATH", "SYSTEMROOT", "SystemRoot", "COMSPEC", "TEMP", "TMP", "PATHEXT", "PROCESSOR_ARCHITECTURE", "LANG", "PSModulePath")
        self.env = {k: v for k, v in os.environ.items() if k in keep}
        self.env.update(HOME=str(self.home), USERPROFILE=str(self.home), LOCALAPPDATA=str(self.home / "AppData" / "Local"),
                        APPDATA=str(self.home / "AppData" / "Roaming"), XDG_RUNTIME_DIR=str(tmp / "run"), PYTHONUTF8="1",
                        DBUS_SESSION_BUS_ADDRESS="unix:path=/nonexistent")
        self.data = {"win32": self.home / "AppData" / "Local" / "hearth",
                     "darwin": self.home / "Library" / "Application Support" / "hearth"}.get(
                         sys.platform, self.home / ".local" / "share" / "hearth")
        self.settings()
        self.pins = self._pins()

    def settings(self):
        """Ports of this test's own; else as the installer writes them."""
        (self.home / ".hearth").mkdir(exist_ok=True)
        (self.home / ".hearth" / "config.yaml").write_text(
            f"bind: {{host: 127.0.0.1, port: {self.port}}}\nbackends:\n  ollama: {{type: ollama, base_url: "
            f"'http://127.0.0.1:{self.ollama_port}', models_dir: '{(self.data / 'models').as_posix()}'}}\n", encoding="utf-8")

    def _pins(self) -> Path:
        pkg = self.tmp / "fake-ollama"
        (pkg).mkdir()
        (pkg / "fake_ollama.py").write_text(FAKE_OLLAMA)
        (pkg / "pyproject.toml").write_text('[project]\nname="fake-ollama"\nversion="1"\n[project.scripts]\nollama="fake_ollama:main"\n'
                                            '[build-system]\nrequires=["setuptools"]\nbuild-backend="setuptools.build_meta"\n'
                                            '[tool.setuptools]\npy-modules=["fake_ollama"]\n')
        env = self.tmp / "fake-env"
        _run(["uv", "venv", "-q", env])
        _run(["uv", "pip", "install", "-q", "--python", env, pkg])
        out = self.releases / "ollama"
        out.mkdir()
        if WIN:
            archive = out / "ollama-windows.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.write(env / "Scripts" / "ollama.exe", "ollama.exe")
        else:
            archive = out / "ollama.tgz"
            with tarfile.open(archive, "w:gz") as t:
                t.add(env / "bin" / "ollama", "bin/ollama")
        real = json.loads((ROOT / "installer" / "pins.json").read_text())
        entry = [{"url": f"{self.url}/ollama/{archive.name}", "sha256": _sha(archive), "size": archive.stat().st_size}]
        real["ollama"] = {"version": "0.0.1", "archives": {k: entry for k in ("linux-x86_64", "windows-x86_64", "macos-arm64")}}
        p = self.tmp / "pins.json"
        p.write_text(json.dumps(real))
        return p

    def release(self, version: str, change=None) -> str:
        """Build a release of this tree as `version`; its manifest's URL."""
        src = self.tmp / f"src-{version}"
        shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns(".git", ".venv*", "__pycache__", "vendor", "*.egg-info", "build", "dist", "wheel*", "hearth-home"))
        for name in ("hearth", "mantel"):
            f = src / "packages" / name / "src" / name / "__init__.py"
            f.write_text(re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', f.read_text(encoding="utf-8")), encoding="utf-8")
        if change:
            change(src)
        out = self.releases / f"v{version}"
        _run([sys.executable, src / "tools" / "release.py", out, "--base-url", f"{self.url}/v{version}/", "--pins", self.pins,
              "--no-extras"], env={**os.environ, "PYTHONUTF8": "1"})
        return f"{self.url}/v{version}/manifest.json"

    def install(self, version: str, *args, ok=True):
        d = self.releases / f"v{version}"
        if WIN:
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                   f"& ([scriptblock]::Create((Get-Content -Raw '{d / 'install.ps1'}'))) {' '.join(args)}"]
        else:
            cmd = ["sh", d / "install.sh", *args]
        return _run(cmd, env=self.env, ok=ok)

    def hearth(self, *args, ok=True, **env):
        return _run([self.data / "bin" / ("hearth.exe" if WIN else "hearth"), *args], env={**self.env, **env}, ok=ok)

    def version(self) -> str:
        try:
            return httpx.get(f"http://127.0.0.1:{self.port}/admin/version", timeout=5).json()["version"]
        except httpx.HTTPError:
            return ""

    def processes(self) -> list[str]:
        out = []
        for p in psutil.process_iter():
            try:
                if any(str(self.data) in a for a in p.cmdline()):
                    out.append(" ".join(p.cmdline()))
            except psutil.Error:
                continue
        return out

    def autostart(self) -> bool:
        if WIN:
            import winreg

            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
                    return str(self.data) in winreg.QueryValueEx(k, "hearth")[0]
            except OSError:
                return False
        p = self.home / ("Library/LaunchAgents/io.github.yumima.hearth.plist" if sys.platform == "darwin"
                         else ".config/autostart/hearth.desktop")
        return p.exists()

    def close(self):
        if (self.data / "bin").is_dir():
            try:
                self.hearth("stop", ok=None, timeout=60)
            except Exception:
                pass
        for p in psutil.process_iter():
            try:
                if any(str(self.data) in a for a in p.cmdline()) and p.pid != os.getpid():
                    p.kill()
            except psutil.Error:
                continue
        self.server.shutdown()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("w"))
    yield w
    w.close()


def _until(what, wait_s=30.0):
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if what():
            return True
        time.sleep(0.2)
    return False


def test_01_install(world):
    world.release("0.2.0")
    r = world.install("0.2.0", "--no-open")
    assert "Choose a model at" in r.stdout
    assert world.version() == "0.2.0"
    # its own Ollama, in its own folder, with its own models
    ollama = httpx.get(f"http://127.0.0.1:{world.ollama_port}/api/version").json()
    assert Path(ollama["dir"]) == world.data / "models"
    cur = json.loads((world.data / "current.json").read_text())
    assert (cur["version"], cur["ollama"], cur["mantel"], cur["previous"]) == ("0.2.0", "0.0.1", True, None)
    for name in ("hearth", "mantel"):
        p = world.data / "bin" / (name + ".exe" if WIN else name)
        assert p.exists()
        if not WIN:
            assert (world.home / ".local" / "bin" / name).resolve() == p.resolve()
    assert world.autostart()
    # two environments: neither has the other in it
    for name, other in (("hearth", "mantel"), ("mantel", "hearth")):
        py = world.data / "versions" / "0.2.0" / name / ("Scripts/python.exe" if WIN else "bin/python")
        assert _run([py, "-I", "-c", f"import {name}"])
        _run([py, "-I", "-c", f"import {other}"], ok=False)
    assert not (world.data / "downloads").exists() and not (world.data / "cache").exists()
    assert httpx.get(f"http://127.0.0.1:{world.port}/").status_code == 200
    assert "verdict" in world.hearth("plan", "--json").stdout


def test_02_installing_again_repairs(world):
    shutil.rmtree(world.data / "versions" / "0.2.0" / "mantel")
    world.hearth("stop")
    (world.data / "ollama" / "0.0.1" / ".complete").unlink()
    world.install("0.2.0", "--no-open")
    assert world.version() == "0.2.0" and (world.data / "versions" / "0.2.0" / "mantel" / ".complete").exists()


def test_03_update(world):
    url = world.release("0.3.0")
    assert "0.3.0 is available" in world.hearth("update", "--check", HEARTH_RELEASE_URL=url).stdout
    assert world.version() == "0.2.0"
    # started as finterm, or a login without systemd, starts it: from its own process
    world.hearth("stop")
    world.hearth("start", "--background")
    assert world.version() == "0.2.0"
    world.hearth("update", "--yes", HEARTH_RELEASE_URL=url)
    assert world.version() == "0.3.0"
    cur = json.loads((world.data / "current.json").read_text())
    assert cur["version"] == "0.3.0" and cur["previous"]["version"] == "0.2.0"
    assert (world.data / "backups" / cur["previous"]["backup"] / "index.json").exists()
    assert (world.data / "versions" / "0.2.0" / "hearth").is_dir()  # to go back to
    assert "is the latest" in world.hearth("update", "--yes", HEARTH_RELEASE_URL=url).stdout


def _breaks_at_start(src: Path):
    f = src / "packages" / "hearth" / "src" / "hearth" / "app.py"
    f.write_text(f.read_text(encoding="utf-8") + "\nraise RuntimeError('this release is broken')\n", encoding="utf-8")


def test_04_an_update_that_does_not_start_goes_back(world):
    url = world.release("0.4.0", _breaks_at_start)
    settings = (world.home / ".hearth" / "config.yaml").read_text()
    r = world.hearth("update", "--yes", ok=False, HEARTH_RELEASE_URL=url)
    assert "0.3.0 is running again" in r.stderr
    assert world.version() == "0.3.0"
    assert json.loads((world.data / "current.json").read_text())["version"] == "0.3.0"
    assert (world.home / ".hearth" / "config.yaml").read_text() == settings


def test_05_a_file_that_is_not_the_published_one_changes_nothing(world):
    url = world.release("0.5.0")
    wheel = next((world.releases / "v0.5.0").glob("hearth-*.whl"))
    wheel.write_bytes(wheel.read_bytes() + b"x")
    r = world.hearth("update", "--yes", ok=False, HEARTH_RELEASE_URL=url)
    assert "not the published file" in r.stderr and world.version() == "0.3.0"


def test_06_back_to_the_version_before(world):
    conf = world.home / ".hearth" / "config.yaml"
    before = conf.read_text()
    conf.write_text(before + "\n# what a newer version wrote\n")
    world.hearth("update", "--rollback", "--yes")
    assert world.version() == "0.2.0" and "newer version" not in conf.read_text()
    assert json.loads((world.data / "current.json").read_text())["version"] == "0.2.0"
    kept = [p for p in (world.data / "backups").glob("*/*config.yaml") if "newer version" in p.read_text()]
    assert kept  # those of then are put aside, not lost
    assert "no earlier version" in world.hearth("update", "--rollback", "--yes", ok=False).stderr


def test_06b_a_newer_script_over_a_running_one_is_an_update(world):
    world.install("0.3.0", "--no-open")
    assert world.version() == "0.3.0"
    prev = json.loads((world.data / "current.json").read_text())["previous"]
    assert prev["version"] == "0.2.0" and (world.data / "backups" / prev["backup"]).is_dir()
    # and one that does not start is taken back, as by `hearth update`
    r = world.install("0.4.0", "--no-open", ok=False)
    assert "0.3.0 is running again" in r.stderr + r.stdout and world.version() == "0.3.0"


def test_06bb_a_script_that_cannot_finish_leaves_the_running_one_running(world):
    url = world.release("0.6.0")
    wheel = next((world.releases / "v0.6.0").glob("mantel-*.whl"))
    wheel.write_bytes(wheel.read_bytes() + b"x")
    r = world.install("0.6.0", "--no-open", ok=False)
    assert "not the published file" in r.stderr + r.stdout and url
    assert world.version() == "0.3.0"


def test_06bd_without_mantel_there_is_no_link_left_to_it(world):
    world.install("0.3.0", "--no-open", "--no-mantel")
    assert world.version() == "0.3.0" and not (world.data / "bin" / ("mantel.exe" if WIN else "mantel")).exists()
    assert not (world.home / ".local" / "bin" / "mantel").is_symlink()
    if sys.platform.startswith("linux"):
        assert not (world.home / ".local" / "share" / "applications" / "mantel.desktop").exists()


def test_06be_a_program_with_a_file_of_it_open_is_not_ended(world):
    f = next((world.data / "versions").glob("*/hearth/pyvenv.cfg"))
    reader = subprocess.Popen([sys.executable, "-c", "import sys, time; open(sys.argv[1]); time.sleep(300)", str(f)])
    try:
        world.hearth("stop")
        world.hearth("start", "--background")
        world.install("0.3.0", "--no-open")  # a repair: stops and starts it
        assert reader.poll() is None and world.version() == "0.3.0"
    finally:
        reader.kill()


def test_06c_it_knows_where_it_is_without_being_told(world):
    """At login there are no variables of the shell it was installed from."""
    py = world.data / "versions" / "0.3.0" / "hearth" / ("Scripts/python.exe" if WIN else "bin/python")
    r = _run([py, "-I", "-c", "from hearth import paths, install; print(paths.data_dir()); print(install.managed())"],
             env={**world.env, "HEARTH_DATA": str(world.tmp / "elsewhere")})
    assert r.stdout.split("\n")[0].strip() == str(world.data) and "True" in r.stdout


def test_07_removing_keeps_what_was_asked_for(world):
    (world.data / "somebody-elses.txt").write_text("not the installer's")
    (world.data / "models").mkdir(exist_ok=True)
    (world.data / "models" / "a-model").write_text("x")
    mine = world.home / "mine.txt"
    mine.write_text("not hearth's")
    world.hearth("uninstall", "--yes", "--keep-models", "--keep-settings")
    assert _until(lambda: sorted(p.name for p in world.data.iterdir()) == ["models", "somebody-elses.txt"], 60)
    (world.data / "somebody-elses.txt").unlink()
    assert (world.home / ".hearth" / "config.yaml").exists() and mine.exists()
    assert not world.autostart() and world.version() == ""
    assert _until(lambda: world.processes() == []), world.processes()


def test_08_installing_beside_kept_models_and_without_mantel(world):
    world.install("0.2.0", "--no-open", "--no-mantel", "--no-autostart")
    assert world.version() == "0.2.0" and not world.autostart()
    assert (world.data / "models" / "a-model").exists()
    assert not (world.data / "bin" / ("mantel.exe" if WIN else "mantel")).exists()
    assert not (world.data / "versions" / "0.2.0" / "mantel").exists()


def test_08b_mantel_can_be_added_later(world):
    url = f"{world.url}/v0.2.0/manifest.json"
    world.hearth("client", "install", HEARTH_RELEASE_URL=url)
    assert (world.data / "bin" / ("mantel.exe" if WIN else "mantel")).exists()
    assert json.loads((world.data / "current.json").read_text())["mantel"] is True
    if sys.platform.startswith("linux"):
        entry = (world.home / ".local" / "share" / "applications" / "mantel.desktop").read_text()
        assert str(world.data / "bin" / "mantel") in entry
    world.hearth("client", "uninstall")
    world.hearth("client", "install", HEARTH_RELEASE_URL=f"{world.url}/nothing")  # it is here: nothing is fetched
    world.hearth("client", "install", "--build", HEARTH_RELEASE_URL=url)  # with hearth running
    assert world.version() == "0.2.0"
    assert _run([world.data / "bin" / ("mantel.exe" if WIN else "mantel"), "--help"], env=world.env).returncode == 0


def test_09_removing_leaves_nothing(world):
    world.hearth("uninstall", "--yes")
    assert _until(lambda: not world.data.exists(), 60), list(world.data.rglob("*"))[:10]
    assert not (world.home / ".hearth").exists()
    assert not any((world.home / ".local" / "bin" / n).is_symlink() for n in ("hearth", "mantel"))
    assert _until(lambda: world.processes() == []), world.processes()
    assert (world.home / "mine.txt").exists()


@pytest.mark.skipif(WIN, reason="links are not used there")
def test_10_somebody_elses_hearth_is_not_replaced(world):
    world.settings()
    link = world.home / ".local" / "bin" / "hearth"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to("/opt/checkout/.venv/bin/hearth")
    r = world.install("0.2.0", "--no-open", ok=False)
    assert "is another hearth (/opt/checkout/.venv/bin/hearth)" in r.stderr
    assert os.readlink(link) == "/opt/checkout/.venv/bin/hearth"
    assert not world.data.exists() and not world.autostart()
    link.unlink()


def test_10b_another_hearth_that_is_running_is_not_replaced(world):
    """Found only by hearth itself, after it was brought: nothing of that stays."""
    world.settings()
    other = subprocess.Popen([sys.executable, "-c", (
        "import http.server, sys\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    server_version = 'hearth/9'\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')\n"
        "    def log_message(self, *_): pass\n"
        "import socketserver\n"
        "socketserver.ThreadingTCPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n"), str(world.port)])
    try:
        assert _until(lambda: _answers(world.port))
        r = world.install("0.2.0", "--no-open", ok=False)
        assert "another hearth is running" in r.stderr + r.stdout
        assert other.poll() is None and not world.data.exists() and not world.autostart()
    finally:
        other.kill()


def _answers(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def test_11_a_folder_that_is_not_ours_is_not_used(world):
    shutil.rmtree(world.data, ignore_errors=True)
    for name in ("thesis.txt", ".hidden", "versions"):
        world.data.mkdir(parents=True)
        (world.data / name).mkdir() if name == "versions" else (world.data / name).write_text("mine")
        r = world.install("0.2.0", "--no-open", ok=False)
        assert "was not made by this installer" in r.stderr + r.stdout
        assert [p.name for p in world.data.iterdir()] == [name]
        shutil.rmtree(world.data)
