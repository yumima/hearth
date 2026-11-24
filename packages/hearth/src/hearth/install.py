"""Installing, updating, rolling back and removing hearth itself.

Everything is in one folder (paths.data_dir()):

    bin/            hearth, mantel, uv — stable names for what is current
    versions/V/     hearth/ and mantel/, an environment each
    python/         the Python they run on
    ollama/W/       the Ollama version V was tested with
    models/ voices/ what the person chose to download
    backups/        settings and chats, from before each update
    current.json    which version is current, and which one before it

A version is built beside the one in use and becomes current in one step, so an
update that fails changes nothing, and the one before can come back. Outside the
folder there are only ~/.hearth (settings, logs), a login entry, mantel's menu
entry and two links in ~/.local/bin; each is removed only if it is ours.

The install scripts bring uv and the first environment, then run this file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import webbrowser
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import urljoin, urlsplit

import httpx

from . import __version__, background, hardware, paths, procs, service
from . import config as cfgmod
from .provision import Problem, fetch

RELEASES = "https://github.com/yumima/hearth/releases"
MARKER = ".hearth-install"
OLLAMA_PORT = 11436  # its own, beside an Ollama the person may have on 11434
_WIN = sys.platform == "win32"
# What can go wrong on a person's computer, said in a line rather than a traceback.
ERRORS = (OSError, ValueError, KeyError, sqlite3.Error, tarfile.TarError, zipfile.BadZipFile, subprocess.SubprocessError)


def root() -> Path:
    return paths.data_dir()


def exe(name: str) -> str:
    return name + ".exe" if _WIN else name


def env_bin(env: Path, name: str) -> Path:
    return env / ("Scripts" if _WIN else "bin") / exe(name)


# ── what a release is ─────────────────────────────────────────────────────────

@dataclass
class Release:
    """manifest.json of a release: every file of it, with its hash."""

    data: dict
    base: str  # file names are relative to it

    @property
    def version(self) -> str:
        return self.data["version"]

    def file(self, name: str) -> tuple[str, str]:
        f = self.data["files"][name]
        return urljoin(self.base, f["name"]), f["sha256"]

    def ollama(self, hw: hardware.Hardware) -> tuple[str, list[dict]]:
        """(version, archives) for this computer."""
        o = self.data["ollama"]
        key = f"{hw.system}-{hw.machine}"
        archives = list(o["archives"].get(key) or [])
        if not archives:
            raise Problem(f"There is no hearth for {key} yet.")
        g = hw.best_gpu()
        if g is not None and g.vendor == "amd":
            archives += o["archives"].get(key + "-amd") or []
        return o["version"], archives


def release(url: str = "", digest: str = "") -> Release:
    """The release at `url` (its manifest), or the latest one. Read over HTTPS
    from the project's own releases; `digest` pins it further."""
    url = url or os.environ.get("HEARTH_RELEASE_URL") or f"{RELEASES}/latest/download/manifest.json"
    parts = urlsplit(url)
    if parts.scheme != "https" and parts.hostname not in ("127.0.0.1", "localhost"):
        raise Problem("A release is only read over HTTPS.")
    r = httpx.get(url, follow_redirects=True, timeout=30.0)
    if r.status_code != 200:
        raise Problem(f"The release could not be read (HTTP {r.status_code}).")
    import hashlib

    if digest and hashlib.sha256(r.content).hexdigest() != digest:
        raise Problem("The release is not the one this installer is for.")
    data = r.json()
    # Its other files are beside it. Not where the download was redirected to
    # (GitHub keeps each file elsewhere), and for "latest", under its own tag: what
    # is latest may be another by the next request.
    base = url.replace("/releases/latest/download/", f"/releases/download/v{data.get('version')}/")
    if data.get("schema") != 1:
        raise Problem("This release needs a newer installer — get hearth again from its page.")
    return Release(data, base)


def version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return ()


# ── what is installed ─────────────────────────────────────────────────────────

def current() -> dict:
    return paths.current()


def _set_current(d: dict) -> None:
    p = root() / "current.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, p)


def managed() -> bool:
    """Is this hearth one the installer made (and not a source checkout, a pip
    install, or an unpacked copy somebody runs)?"""
    if not (root() / MARKER).is_file():
        return False
    try:
        return Path(sys.prefix).resolve().is_relative_to((root() / "versions").resolve())
    except OSError:
        return False


def _inside(p: Path) -> bool:
    try:
        return Path(os.path.realpath(p)).is_relative_to(os.path.realpath(root()))
    except OSError:
        return False


def conflicts() -> list[str]:
    """What is already here that is somebody's own: not replaced, not changed."""
    out = []
    if f := service.foreign():
        out.append(f"{f} starts a hearth that this installer did not make")
    for name in ("hearth", "mantel"):
        link = Path.home() / ".local" / "bin" / name
        if not _WIN and (link.is_symlink() or link.exists()) and not _inside(link):
            out.append(f"{link} is another {name}" + (f" ({os.path.realpath(link)})" if link.is_symlink() else ""))
    cfg = cfgmod.load(create=False)
    if background.gateway_identity(cfg) is not None:
        pid = procs.listening_on(cfg.bind_port)
        argv = procs.argv(pid) if pid else []
        if not any(_inside(Path(a)) for a in argv[:3] if os.path.isabs(a)):
            out.append(f"another hearth is running (port {cfg.bind_port}{f', pid {pid}' if pid else ''})")
    return out


# ── building a version ────────────────────────────────────────────────────────

Say = Callable[[str], None]


def _uv() -> str:
    p = root() / "bin" / exe("uv")
    if p.is_file():
        return str(p)
    if found := shutil.which("uv"):
        return found
    raise Problem("uv, which installs hearth's parts, is missing — run the installer again.")


def _run(cmd: list[str], what: str, env: dict | None = None) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})}, check=False)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-6:])
        raise Problem(f"{what} failed:\n{tail}")


def _uv_env() -> dict:
    # Nothing of the person's own Python setup has a part in it.
    return {"UV_PYTHON_INSTALL_DIR": str(root() / "python"), "UV_CACHE_DIR": str(root() / "cache"),
            "UV_NO_CONFIG": "1", "UV_PYTHON_PREFERENCE": "only-managed", "UV_LINK_MODE": "copy",
            "VIRTUAL_ENV": "", "PYTHONPATH": "", "PIP_INDEX_URL": "", "UV_INDEX_URL": "", "UV_EXTRA_INDEX_URL": "",
            "UV_DEFAULT_INDEX": "https://pypi.org/simple"}


def build_env(rel: Release, name: str, say: Say, replacing: Callable[[], None] = lambda: None, again: bool = False) -> Path:
    """versions/V/name, complete, or it is made again. `replacing` is called before
    something that is there, and may be in use, is moved away; if making it fails,
    that is put back."""
    env = root() / "versions" / rel.version / name
    done = env / ".complete"
    if done.is_file() and not again:
        return env
    say(f"Installing {name} {rel.version}…")
    downloads = root() / "downloads" / rel.version
    files = {}
    for kind in ("wheel", "requirements"):
        url, digest = rel.file(f"{name}-{kind}")
        files[kind] = downloads / url.rsplit("/", 1)[-1]
        fetch(url, files[kind], digest)
    before = env.with_name(name + ".before")
    if before.exists() and not done.is_file():  # the last time was cut short: that is the one that works
        shutil.rmtree(env, ignore_errors=True)
        before.rename(env)
        if not again:
            return env
    shutil.rmtree(before, ignore_errors=True)
    uv, env_vars = _uv(), _uv_env()
    if env.exists():
        replacing()
    try:
        if done.is_file():
            env.rename(before)
        elif env.exists():
            shutil.rmtree(env)
        _run([uv, "python", "install", "--quiet", rel.data["python"]], "Installing Python", env_vars)
        _run([uv, "venv", "--quiet", "--python", rel.data["python"], str(env)], f"Making {name}'s environment", env_vars)
        _run([uv, "pip", "install", "--quiet", "--python", str(env), "--require-hashes", "--only-binary", ":all:",
              "-r", str(files["requirements"])], f"Installing what {name} needs", env_vars)
        _run([uv, "pip", "install", "--quiet", "--python", str(env), "--no-deps", "--no-index", str(files["wheel"])],
             f"Installing {name}", env_vars)
        _run([str(env_bin(env, "python")), "-I", "-c",
              f"import {name}; assert {name}.__version__ == {rel.version!r}, {name}.__version__"], f"Starting {name}")
        done.write_text("")
    except BaseException:
        _put_back(before, env)
        raise
    shutil.rmtree(before, ignore_errors=True)
    return env


def _put_back(before: Path, env: Path) -> None:
    """(What cannot be put back now — a file held open — is, by the next build_env.)"""
    if not before.exists():
        return
    try:
        if env.exists():
            shutil.rmtree(env, ignore_errors=True)
        if env.exists():
            (env / ".complete").unlink(missing_ok=True)
            env.rename(env.with_name(f"{env.name}.{time.time_ns()}.failed"))
        before.rename(env)
    except OSError:
        pass


def _safe_members(names: list[str]) -> None:
    for n in names:
        p = PurePosixPath(n.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts or ":" in n:
            raise Problem("an archive names a file outside its folder")


def unpack(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            _safe_members(z.namelist())
            z.extractall(dest)
    elif archive.name.endswith(".zst"):
        import zstandard

        with open(archive, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as s, \
                tarfile.open(fileobj=s, mode="r|") as t:
            t.extractall(dest, filter="data")
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")


def get_ollama(rel: Release, hw: hardware.Hardware, say: Say, replacing: Callable[[], None] = lambda: None) -> str:
    version, archives = rel.ollama(hw)
    dest = root() / "ollama" / version
    done = dest / ".complete"
    want = " ".join(a["sha256"] for a in archives)
    if done.is_file() and done.read_text() == want:
        return version
    size = sum(a["size"] for a in archives)
    free = hardware._disk_free(root())
    if free is not None and free * (1 << 20) < size * 3.5:
        raise Problem(f"Not enough free disk space: the engine needs {size * 3.5 / 1e9:.0f} GB while it is installed, "
                      f"and {free / 1024:.1f} GB are free.")
    say(f"Downloading the engine (Ollama {version}, {size / 1e9:.1f} GB)…")
    new = dest.with_name(dest.name + ".new")
    if new.exists():
        shutil.rmtree(new)
    for a in archives:
        f = root() / "downloads" / a["url"].rsplit("/", 1)[-1]
        fetch(a["url"], f, a["sha256"], _Progress(say, a["size"]))
    new.mkdir(parents=True)
    say("Unpacking it…")
    for a in archives:
        f = root() / "downloads" / a["url"].rsplit("/", 1)[-1]
        unpack(f, new)
        f.unlink()
    if dest.exists():
        replacing()
        shutil.rmtree(dest)
    new.rename(dest)
    done.write_text(want)
    return version


class _Progress:
    """What fetch() reports to: every tenth of a download is said."""

    def __init__(self, say: Say, size: int):
        import threading

        self._say, self._size, self._said = say, size, -1
        self.total, self._done, self.stop = 0, 0, threading.Event()

    @property
    def done(self) -> int:
        return self._done

    @done.setter
    def done(self, n: int) -> None:
        self._done = n
        tenth = n * 10 // max(1, self._size)
        if tenth > self._said:
            self._said = tenth
            self._say(f"  {min(100, tenth * 10)}%")


# ── making a version the current one ──────────────────────────────────────────

def link_programs(version: str, mantel: bool) -> None:
    bin_dir = root() / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hearth", "mantel", *(("hearthw",) if _WIN else ())):
        target = env_bin(root() / "versions" / version / name.removesuffix("w"), name)
        link = bin_dir / exe(name)
        if name == "mantel" and not mantel:
            if link.exists():
                _mantel("uninstall")
            user = Path.home() / ".local" / "bin" / name
            if not _WIN and user.is_symlink() and _inside(user):
                user.unlink()
            link.unlink(missing_ok=True)
            continue
        new = not link.exists()
        tmp = bin_dir / (exe(name) + ".tmp")
        tmp.unlink(missing_ok=True)
        if _WIN:
            # The launcher names its Python by its full path, so a copy works. One
            # that is running cannot be replaced, and can be moved out of the way.
            if link.is_file() and link.read_bytes() == target.read_bytes():
                continue
            shutil.copy2(target, tmp)
            for old in bin_dir.glob(exe(name) + ".*.old"):
                try:
                    old.unlink()
                except OSError:
                    pass  # still running
            if link.exists():
                n = next(n for n in range(1000) if not (bin_dir / f"{exe(name)}.{n}.old").exists())
                link.rename(bin_dir / f"{exe(name)}.{n}.old")
        else:
            tmp.symlink_to(os.path.relpath(target, bin_dir))
        os.replace(tmp, link)
        if not _WIN:
            user = Path.home() / ".local" / "bin" / name
            user.parent.mkdir(parents=True, exist_ok=True)
            if user.is_symlink() and _inside(user):
                user.unlink()
            if not (user.exists() or user.is_symlink()):
                user.symlink_to(link)
        if name == "mantel" and new:
            _mantel("install")  # (a version without it took it out)


def first_config() -> None:
    """An Ollama of its own, and no model chosen yet. Settings that are there stay."""
    p = cfgmod.config_path()
    if p.exists():
        return
    cfg = cfgmod.Config(backends={"ollama": cfgmod.Backend(
        "ollama", "ollama", f"http://127.0.0.1:{OLLAMA_PORT}", models_dir=str(root() / "models"))}, path=p)
    cfgmod.save(cfg)


def _own_processes() -> list:
    """Programs of this installation that are running — what they run is in the
    folder: its Python, its Ollama, its launchers — and what they have started.
    Not the one that asks, nor its launcher; not one that has a file of it open."""
    import psutil

    me = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    bases = {os.path.abspath(root()) + os.sep, os.path.realpath(root()) + os.sep}
    out: dict[int, psutil.Process] = {}
    for p in psutil.process_iter():
        try:
            argv = p.cmdline()
            programs = [p.exe(), *(argv[:1] if argv and os.path.isabs(argv[0]) else [])]
            if any(a.startswith(b) for a in programs if a for b in bases):
                for q in (p, *p.children(recursive=True)):
                    out[q.pid] = q
        except psutil.Error:
            continue
    return [p for pid, p in out.items() if pid not in me]


def _running() -> bool:
    cfg = cfgmod.load(create=False)
    return background.gateway_identity(cfg) is not None and procs.listening_on(cfg.bind_port) in {p.pid for p in _own_processes()}


def _stop_everything() -> None:
    """The gateway in order (its Ollama with it), then whatever is left of ours —
    asked of the processes, not of a version that may not be able to answer."""
    if not _WIN and service.unit_path().exists() and not service.foreign():
        service.systemctl("stop", service.UNIT_NAME)
    try:
        cfg = cfgmod.load(create=False)
        pid = procs.listening_on(cfg.bind_port)
        if pid in {p.pid for p in _own_processes()}:
            try:
                httpx.post(f"http://{cfg.bind_host}:{cfg.bind_port}/admin/shutdown", timeout=3.0)
            except httpx.HTTPError:
                pass  # busy: it may have heard
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and procs.alive(pid):
                time.sleep(0.2)
    except Exception:  # settings that cannot be read, a gateway that does not answer: it is ended all the same
        pass
    for p in _own_processes():
        procs.end_tree(p.pid)


def _start(version: str) -> tuple[bool, str]:
    """Started, and it is this version that answers."""
    cfg = cfgmod.load(create=False)
    program = str(root() / "bin" / exe("hearth"))
    svc = None
    if sys.platform.startswith("linux") and service.unit_path().exists() and not service.foreign() and service.has_systemd():
        svc = background.Service(lambda: service.systemctl("start", service.UNIT_NAME),
                                 lambda: service.systemctl("is-active", "--quiet", service.UNIT_NAME) in (0, 3))
    ok, message = background.start(cfg, [program, "start"], svc)
    if ok and not _serving(version):
        return False, "another version answers"
    return ok, message


def _serving(version: str) -> bool:
    cfg = cfgmod.load(create=False)
    try:
        r = httpx.get(f"http://{cfg.bind_host}:{cfg.bind_port}/admin/version", timeout=5.0)
        return r.json().get("version") == version
    except (httpx.HTTPError, ValueError):
        return False


def finish(rel: Release, mantel: bool, autostart: bool, start: bool, open_page: bool, say: Say) -> None:
    """The rest of an installation, run from its own new environment. Over an
    earlier one it is an update; over the same version, a repair. One that is
    running goes on until everything is here, and runs again afterwards."""
    (root() / MARKER).write_text("hearth's installer made this folder, and removing hearth removes it.\n")
    old = current()
    run = bool(old) and _running()
    stopped = []

    def replacing() -> None:
        if not stopped:
            stopped.append(1)
            _stop_everything()

    try:
        build_env(rel, "hearth", say, replacing)
        if mantel:
            build_env(rel, "mantel", say, replacing)
        ollama = get_ollama(rel, hardware.probe(), say, replacing)
        new = {"version": rel.version, "ollama": ollama, "mantel": mantel}
        first_config()
        if old and old.get("version") != rel.version:
            _switch(new, old, run or start, say)
        else:
            if old:
                replacing()
            _set_current({**new, "previous": old.get("previous")})
            link_programs(rel.version, mantel)
        if mantel:
            _mantel("install")
        if autostart:
            try:
                service.enable(str(root() / "bin" / exe("hearthw" if _WIN else "hearth")))
            except (service.Foreign, *ERRORS) as e:
                say(f"hearth will not start at login: {e}")
        _tidy()
    except BaseException:
        if stopped and run:
            try:
                _start(current()["version"])  # it was running
            except Exception as e:
                say(f"hearth was not started again: {str(e) or type(e).__name__}")
        raise
    if not (start or run):
        return
    ok, message = _start(rel.version)
    if not ok:
        raise Problem(f"hearth is installed, and did not start: {message}")
    say(f"hearth is {message}")
    cfg = cfgmod.load(create=False)
    url = f"http://{cfg.bind_host}:{cfg.bind_port}/"
    say(f"Choose a model at {url}")
    if open_page and start:
        webbrowser.open(url)


def add_mantel(rebuild: bool, say: Say) -> None:
    """mantel, for an installation made without it (or made again)."""
    cur = current()
    version = cur["version"]
    rel = release(os.environ.get("HEARTH_RELEASE_URL") or f"{RELEASES}/download/v{version}/manifest.json")
    if rel.version != version:
        raise Problem(f"That release is {rel.version}, and hearth here is {version}: update hearth first.")
    env = root() / "versions" / version / "mantel"
    prefixes = {os.path.abspath(env) + os.sep, os.path.realpath(env) + os.sep}

    def running() -> list[int]:
        import psutil

        out = []
        for p in _own_processes():
            try:
                if any(a.startswith(b) for a in (p.exe(), *p.cmdline()[:2]) if a for b in prefixes):
                    out.append(p.pid)
            except psutil.Error:
                continue
        return out

    build_env(rel, "mantel", say, lambda: [procs.end_tree(pid) for pid in running()] and None, again=rebuild)
    _set_current({**cur, "mantel": True})
    link_programs(version, True)
    shutil.rmtree(root() / "downloads", ignore_errors=True)
    shutil.rmtree(root() / "cache", ignore_errors=True)


def _mantel(action: str) -> None:
    """Its entry in the app menu. Nothing that goes wrong there stops an installation."""
    p = root() / "bin" / exe("mantel")
    if p.exists():
        try:
            subprocess.run([str(p), action], capture_output=True, check=False, timeout=60,
                           env={**os.environ, "MANTEL_PROGRAM": str(p)})
        except (OSError, subprocess.SubprocessError):
            pass


def _tidy() -> None:
    """Versions, engines and downloads nothing needs any more."""
    cur = current()
    prev = cur.get("previous") or {}
    versions = {cur.get("version", ""), prev.get("version", "")}
    engines = {cur.get("ollama", ""), prev.get("ollama", "")}
    for folder, wanted in (("versions", versions), ("ollama", engines)):
        d = root() / folder
        for p in d.iterdir() if d.is_dir() else []:
            if p.name not in wanted:
                shutil.rmtree(p, ignore_errors=True)
    for name in ("downloads", "cache"):
        shutil.rmtree(root() / name, ignore_errors=True)
    backups = sorted((root() / "backups").iterdir()) if (root() / "backups").is_dir() else []
    for p in backups[:-5]:
        if p.name != prev.get("backup"):
            shutil.rmtree(p, ignore_errors=True)


# ── update, and back ──────────────────────────────────────────────────────────

def _mantel_files() -> list[Path]:
    data = Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")) / "mantel"
    conf = Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")) / "mantel"
    return [*sorted(data.glob("*.db")), conf / "config.yaml"]


def backup() -> Path:
    """Settings and chats as they are, for a version that cannot read what a newer
    one wrote. A database is copied as its own program would, not as a file."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in range(100):
        dest = root() / "backups" / (stamp + (f"-{n}" if n else ""))
        try:
            dest.mkdir(parents=True)
            break
        except FileExistsError:
            continue
    files = [cfgmod.config_path(), paths.home() / "measured.json", *_mantel_files()]
    index = {}
    for i, f in enumerate(f for f in files if f.is_file()):
        copy = dest / f"{i}-{f.name}"
        if f.suffix == ".db":
            with sqlite3.connect(f) as src, sqlite3.connect(copy) as dst:
                src.backup(dst)
            src.close()
            dst.close()
        else:
            shutil.copy2(f, copy)
        index[copy.name] = str(f)
    (dest / "index.json").write_text(json.dumps(index))
    return dest


def restore(saved: Path) -> None:
    for name, where in json.loads((saved / "index.json").read_text()).items():
        target = Path(where)
        for extra in (target.with_name(target.name + s) for s in ("-wal", "-shm")):
            extra.unlink(missing_ok=True)
        tmp = target.with_name(target.name + ".restoring")
        shutil.copy2(saved / name, tmp)
        os.replace(tmp, target)


def available() -> Release | None:
    """A newer release than the current one, if there is one."""
    rel = release()
    return rel if version_tuple(rel.version) > version_tuple(current().get("version") or __version__) else None


def _switch(new: dict, old: dict, run: bool, say: Say) -> None:
    """From one version to another, or — whatever goes wrong — not at all."""
    saved, changed = None, False
    try:
        _stop_everything()
        saved = backup()
        changed = True
        _set_current({**new, "previous": {**{k: old[k] for k in ("version", "ollama")}, "mantel": bool(old.get("mantel")),
                                          "backup": saved.name}})
        link_programs(new["version"], bool(new.get("mantel")))
        say("Starting it…")
        ok, message = _start(new["version"])
        if not ok:
            raise Problem(f"{new['version']} did not start ({message})")
    except BaseException as e:
        say(f"{str(e) or type(e).__name__}. Going back to {old['version']}…")
        ok, message = _back(old, saved, run, changed)
        if not isinstance(e, Exception):
            raise
        raise Problem(f"The update did not work. {old['version']} is " + (
            f"current, with this trouble: {message}" if not ok else "running again." if run else "back.")) from e
    if not run:
        _stop_everything()


def _back(old: dict, saved: Path | None, run: bool, changed: bool = True) -> tuple[bool, str]:
    """As it was, as far as it goes: no step that fails keeps the next from being
    tried. (nothing went wrong, what did)."""
    trouble = []

    def step(f, *args) -> None:
        try:
            f(*args)
        except Exception as e:
            trouble.append(f"{str(e) or type(e).__name__}")

    step(_stop_everything)
    if changed:
        if saved is not None and (saved / "index.json").is_file():
            step(restore, saved)  # before anything reads what the other version wrote
        step(_set_current, old)
        step(link_programs, old["version"], bool(old.get("mantel")))
    if run:
        def start() -> None:
            ok, message = _start(old["version"])
            if not ok:
                raise Problem(message)
        step(start)
    return not trouble, "; ".join(trouble)


def update(rel: Release, say: Say) -> None:
    if not managed():
        raise Problem("This hearth was not made by its installer (a source checkout?): update it where it came from.")
    old = current()
    mantel = bool(old.get("mantel"))
    build_env(rel, "hearth", say)          # beside the one in use: what fails here changes nothing
    if mantel:
        build_env(rel, "mantel", say)
    ollama = get_ollama(rel, hardware.probe(), say)
    run = _running()
    _switch({"version": rel.version, "ollama": ollama, "mantel": mantel}, old, run, say)
    _tidy()
    say(f"hearth {rel.version} is ready.")


def rollback_to() -> dict:
    """The version before, if it can be gone back to."""
    prev = current().get("previous")
    if not managed() or not prev:
        raise Problem("There is no earlier version to go back to.")
    if not (root() / "versions" / prev["version"] / "hearth" / ".complete").is_file():
        raise Problem(f"{prev['version']} is no longer here.")
    return prev


def rollback(say: Say) -> None:
    """The version before, with the settings and chats of then: those of now are
    put aside, not lost."""
    prev, cur = rollback_to(), current()
    run = _running()
    say(f"Going back to {prev['version']}…")
    now, changed = None, False
    try:
        _stop_everything()
        now = backup()
        saved = root() / "backups" / prev.get("backup", "-")
        changed = True
        ok, message = _back({k: v for k, v in prev.items() if k != "backup"}, saved if saved.is_dir() else None, run)
        if not ok:
            raise Problem(f"{prev['version']} could not be gone back to ({message})")
    except BaseException as e:
        ok, message = _back(cur, now, run, changed)
        if not isinstance(e, Exception):
            raise
        raise Problem(f"{str(e) or type(e).__name__}. {cur['version']} is " + (
            f"current, with this trouble: {message}" if not ok else "running again." if run else "current.")) from e
    say(f"hearth {prev['version']} is current. The settings and chats of {cur['version']} are kept in {now}.")


# ── removing ──────────────────────────────────────────────────────────────────

# What the installer makes; nothing else in the folder is removed. What it takes to
# run it again comes last, and only when the rest is gone.
_PARTS = ("ollama", "downloads", "cache", "backups", "current.tmp")
_LAST = ("versions", "python", "bin", "current.json")


def uninstall(keep_models: bool, keep_settings: bool, say: Say) -> None:
    base = root()
    if not (base / MARKER).is_file():
        raise Problem(f"{base} was not made by hearth's installer; nothing was removed.")
    if base in (Path.home(), Path(base.anchor)) or len(base.parts) < 3:
        raise Problem(f"{base} cannot be hearth's folder; nothing was removed.")
    _stop_everything()
    service.disable(remove=True)
    _mantel("uninstall")
    for name in ("hearth", "mantel"):
        link = Path.home() / ".local" / "bin" / name
        if not _WIN and link.is_symlink() and _inside(link):
            link.unlink()
    if not keep_settings:
        for f in ("config.yaml", "measured.json", "hearth.log", "hearth.log.1", "ollama.log", "ollama.log.1",
                  "hearth.pid", "ollama.pid"):
            (paths.home() / f).unlink(missing_ok=True)
        try:
            paths.home().rmdir()  # only if nothing else is in it
        except OSError:
            pass
    parts = (*(() if keep_models else ("models", "voices")), *_PARTS, *_LAST)
    doomed = [base / p for p in parts if (base / p).exists()]
    kept = (f" Models and voices are kept in {base}." if keep_models else "") \
        + " mantel's chats and settings are kept where they were."
    if _WIN:
        _remove_later(base, doomed)
        say("hearth is being removed; that takes a few seconds." + kept)
        return
    left = []
    for p in doomed:
        if left and p.name in _LAST:
            continue
        try:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
        except OSError as e:
            left.append(f"{p} ({e.strerror or e})")
    if left:
        raise Problem("Not all of it could be removed: " + "; ".join(left) + f". Then run {base / 'bin' / 'hearth'} uninstall again.")
    (base / MARKER).unlink()
    try:
        base.rmdir()
    except OSError:
        pass  # what was kept, or somebody's own
    say("hearth is removed." + kept)


def _remove_later(base: Path, doomed: list[Path]) -> None:
    """On Windows the program that removes is one of the things removed: a script
    does it once this has ended, trying for a while, the marker last and only when
    the rest is gone."""
    import tempfile

    q = [str(p).replace("%", "%%") for p in doomed]
    b = str(base).replace("%", "%%")
    lines = ["@echo off", "chcp 65001 >nul", "set n=0", ":again", "ping -n 2 127.0.0.1 >nul", "set /a n+=1"]
    for p, quoted in zip(doomed, q):  # what it takes to run it again comes last, when the rest is gone
        lines += [f'if exist "{quoted}\\" rmdir /s /q "{quoted}" 2>nul', f'if exist "{quoted}" del /f /q "{quoted}" 2>nul',
                  f'if exist "{quoted}" if %n% lss 30 goto again', f'if exist "{quoted}" goto end']
    lines += [f'del /f /q "{b}\\{MARKER}"', f'rmdir "{b}" 2>nul', ":end", '(goto) 2>nul & del "%~f0"']
    fd, name = tempfile.mkstemp(prefix="hearth-remove-", suffix=".cmd")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    for leave_job in (0x01000000, 0):  # out of the job of the window it was asked in, where that is allowed
        try:
            subprocess.Popen(f'cmd /d /s /c ""{name}""', cwd=str(Path.home()), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x08000000 | leave_job)
            return
        except OSError:
            if not leave_job:
                os.unlink(name)
                raise


# ── command line (for the install scripts) ────────────────────────────────────

def say(text: str) -> None:
    """Nobody reading (a window that was closed) does not stop what is being done."""
    try:
        print(text, flush=True)
    except OSError:
        pass


def ended_in_order() -> None:
    """Asked to end, or the terminal closed: as with Ctrl-C, a switch is taken back."""
    import signal

    asked = [getattr(signal, n) for n in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, n)]
    asked = [n for n in asked if signal.getsignal(n) != signal.SIG_IGN]  # nohup

    def stop(signum, frame) -> None:
        for n in asked:  # once is enough: going back is not ended, too
            signal.signal(n, signal.SIG_IGN)
        raise KeyboardInterrupt if signum == signal.SIGINT else SystemExit(128 + signum)

    for n in asked:
        signal.signal(n, stop)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hearth.install")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check", help="may hearth be installed here?")
    s = sub.add_parser("finish")
    s.add_argument("--release", required=True)
    s.add_argument("--sha256", default="")
    s.add_argument("--no-mantel", action="store_true")
    s.add_argument("--no-autostart", action="store_true")
    s.add_argument("--no-start", action="store_true")
    s.add_argument("--no-open", action="store_true")
    a = p.parse_args(argv)
    ended_in_order()
    try:
        found = conflicts()
        if found:
            print("hearth was not installed, so as not to disturb what is here:\n  - " + "\n  - ".join(found),
                  file=sys.stderr)
            return 3
        if a.cmd == "finish":
            finish(release(a.release, a.sha256), not a.no_mantel, not a.no_autostart, not a.no_start, not a.no_open,
                   say)
        return 0
    except Problem as e:
        print(e, file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"The connection failed ({e or type(e).__name__}). Run the installer again to go on.", file=sys.stderr)
        return 1
    except ERRORS as e:
        print(f"{type(e).__name__}: {e}. Run the installer again to go on.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
