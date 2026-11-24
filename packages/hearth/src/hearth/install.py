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
    # "latest" redirects to a tag: its other files are there, whatever becomes latest since
    base = str(r.history[0].headers.get("location") or r.url) if r.history else str(r.url)
    if "/releases/latest/" in url and "/releases/download/" not in base:
        base = f"{RELEASES}/download/v{data['version']}/"
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


def build_env(rel: Release, name: str, say: Say) -> Path:
    """versions/V/name, complete, or it is made again."""
    env = root() / "versions" / rel.version / name
    done = env / ".complete"
    if done.is_file():
        return env
    say(f"Installing {name} {rel.version}…")
    if env.exists():
        shutil.rmtree(env)
    downloads = root() / "downloads" / rel.version
    files = {}
    for kind in ("wheel", "requirements"):
        url, digest = rel.file(f"{name}-{kind}")
        files[kind] = downloads / url.rsplit("/", 1)[-1]
        fetch(url, files[kind], digest)
    uv, env_vars = _uv(), _uv_env()
    _run([uv, "python", "install", "--quiet", rel.data["python"]], "Installing Python", env_vars)
    _run([uv, "venv", "--quiet", "--python", rel.data["python"], str(env)], f"Making {name}'s environment", env_vars)
    _run([uv, "pip", "install", "--quiet", "--python", str(env), "--require-hashes", "--only-binary", ":all:",
          "-r", str(files["requirements"])], f"Installing what {name} needs", env_vars)
    _run([uv, "pip", "install", "--quiet", "--python", str(env), "--no-deps", "--no-index", str(files["wheel"])],
         f"Installing {name}", env_vars)
    _run([str(env_bin(env, "python")), "-I", "-c",
          f"import {name}; assert {name}.__version__ == {rel.version!r}, {name}.__version__"], f"Starting {name}")
    done.write_text("")
    return env


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


def get_ollama(rel: Release, hw: hardware.Hardware, say: Say) -> str:
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
    for name in ("hearth", "mantel"):
        target = env_bin(root() / "versions" / version / name, name)
        link = bin_dir / exe(name)
        if name == "mantel" and not mantel:
            link.unlink(missing_ok=True)
            continue
        tmp = bin_dir / (exe(name) + ".tmp")
        tmp.unlink(missing_ok=True)
        if _WIN:
            # The launcher names its Python by its full path, so a copy works. One
            # that is running cannot be replaced, and can be moved out of the way.
            shutil.copy2(target, tmp)
            old = bin_dir / (exe(name) + ".old")
            try:
                old.unlink(missing_ok=True)
                if link.exists():
                    link.rename(old)
            except OSError:
                pass
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


def first_config() -> None:
    """An Ollama of its own, and no model chosen yet. Settings that are there stay."""
    p = cfgmod.config_path()
    if p.exists():
        return
    cfg = cfgmod.Config(backends={"ollama": cfgmod.Backend(
        "ollama", "ollama", f"http://127.0.0.1:{OLLAMA_PORT}", models_dir=str(root() / "models"))}, path=p)
    cfgmod.save(cfg)


def _stop_everything() -> None:
    cfg = cfgmod.load(create=False)
    if not _WIN and service.unit_path().exists():
        service.systemctl("stop", service.UNIT_NAME)
    for name in ("mantel", "hearth"):
        p = root() / "bin" / exe(name)
        if p.exists():
            subprocess.run([str(p), "stop"], capture_output=True, check=False, timeout=60)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and background.gateway_identity(cfg) is not None:
        time.sleep(0.2)


def _start() -> tuple[bool, str]:
    cfg = cfgmod.load(create=False)
    program = str(root() / "bin" / exe("hearth"))
    svc = None
    if sys.platform.startswith("linux") and service.unit_path().exists() and service.has_systemd():
        svc = background.Service(lambda: service.systemctl("start", service.UNIT_NAME),
                                 lambda: service.systemctl("is-active", "--quiet", service.UNIT_NAME) in (0, 3))
    return background.start(cfg, [program, "start"], svc)


def _serving(version: str) -> bool:
    cfg = cfgmod.load(create=False)
    try:
        r = httpx.get(f"http://{cfg.bind_host}:{cfg.bind_port}/admin/version", timeout=5.0)
        return r.json().get("version") == version
    except (httpx.HTTPError, ValueError):
        return False


def finish(rel: Release, mantel: bool, autostart: bool, start: bool, open_page: bool, say: Say) -> None:
    """The rest of a first installation, run from its own new environment."""
    (root() / MARKER).write_text("hearth's installer made this folder, and removing hearth removes it.\n")
    build_env(rel, "hearth", say)
    if mantel:
        build_env(rel, "mantel", say)
    hw = hardware.probe()
    ollama = get_ollama(rel, hw, say)
    before = current()
    _set_current({"version": rel.version, "ollama": ollama, "mantel": mantel,
                  "previous": before if before and before.get("version") != rel.version else before.get("previous")})
    link_programs(rel.version, mantel)
    first_config()
    if mantel:
        _mantel("install")
    if autostart:
        try:
            service.enable(str(root() / "bin" / exe("hearth")))
        except (service.Foreign, OSError) as e:
            say(f"hearth will not start at login: {e}")
    _tidy(keep={rel.version})
    if start:
        ok, message = _start()
        say(f"hearth is {message}" if ok else f"hearth did not start: {message}")
        if not ok:
            raise Problem("It is installed; `hearth start` says why it does not run.")
        cfg = cfgmod.load(create=False)
        url = f"http://{cfg.bind_host}:{cfg.bind_port}/"
        say(f"Choose a model at {url}")
        if open_page:
            webbrowser.open(url)


def _mantel(action: str) -> None:
    p = root() / "bin" / exe("mantel")
    if p.exists():
        subprocess.run([str(p), action], capture_output=True, check=False, timeout=60,
                       env={**os.environ, "MANTEL_PROGRAM": str(p)})


def _tidy(keep: set[str]) -> None:
    """Versions, engines and downloads nothing needs any more."""
    cur = current()
    prev = cur.get("previous") or {}
    versions = keep | {cur.get("version", ""), prev.get("version", "")}
    engines = {cur.get("ollama", ""), prev.get("ollama", "")}
    for folder, wanted in (("versions", versions), ("ollama", engines)):
        d = root() / folder
        for p in d.iterdir() if d.is_dir() else []:
            if p.name not in wanted:
                shutil.rmtree(p, ignore_errors=True)
    for name in ("downloads", "cache"):
        shutil.rmtree(root() / name, ignore_errors=True)
    backups = sorted((root() / "backups").iterdir()) if (root() / "backups").is_dir() else []
    for p in backups[:-3]:
        shutil.rmtree(p, ignore_errors=True)


# ── update, and back ──────────────────────────────────────────────────────────

def _mantel_files() -> list[Path]:
    data = Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")) / "mantel"
    conf = Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")) / "mantel"
    return [*sorted(data.glob("*.db")), conf / "config.yaml"]


def backup() -> Path:
    """Settings and chats as they are, for a version that cannot read what a newer
    one wrote. A database is copied as its own program would, not as a file."""
    dest = root() / "backups" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True)
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


def update(rel: Release, say: Say) -> None:
    if not managed():
        raise Problem("This hearth was not made by its installer (a source checkout?): update it where it came from.")
    old = current()
    mantel = bool(old.get("mantel"))
    build_env(rel, "hearth", say)          # beside the one in use: what fails here changes nothing
    if mantel:
        build_env(rel, "mantel", say)
    ollama = get_ollama(rel, hardware.probe(), say)
    running = background.gateway_identity(cfgmod.load(create=False)) is not None
    _stop_everything()
    saved = backup()
    _set_current({"version": rel.version, "ollama": ollama, "mantel": mantel,
                  "previous": {"version": old["version"], "ollama": old["ollama"], "mantel": mantel, "backup": saved.name}})
    link_programs(rel.version, mantel)
    say("Starting it…")
    ok, message = _start()
    if not (ok and _serving(rel.version)):
        say(f"{rel.version} did not start ({message}). Going back to {old['version']}…")
        _back(old, saved)
        raise Problem(f"The update did not work, and {old['version']} is running again." if running else
                      f"The update did not work, and {old['version']} is back.")
    if not running:
        _stop_everything()
    _tidy(keep={rel.version})
    say(f"hearth {rel.version} is ready.")


def _back(old: dict, saved: Path | None) -> None:
    _stop_everything()
    if saved is not None and (saved / "index.json").is_file():
        restore(saved)
    _set_current(old)
    link_programs(old["version"], bool(old.get("mantel")))
    _start()


def rollback(say: Say) -> None:
    cur = current()
    prev = cur.get("previous")
    if not managed() or not prev:
        raise Problem("There is no earlier version to go back to.")
    if not (root() / "versions" / prev["version"] / "hearth" / ".complete").is_file():
        raise Problem(f"{prev['version']} is no longer here.")
    say(f"Going back to {prev['version']}…")
    saved = root() / "backups" / prev.get("backup", "-")
    keep = {k: v for k, v in prev.items() if k != "backup"}
    _back(keep, saved if saved.is_dir() else None)
    say(f"hearth {prev['version']} is current. Settings and chats are as they were before the update.")


# ── removing ──────────────────────────────────────────────────────────────────

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
    stay = {"models", "voices"} if keep_models else set()
    doomed = [p for p in base.iterdir() if p.name not in stay]
    if _WIN:
        # This program is in there: the rest is done when it has ended.
        cmds = " & ".join((f'rmdir /s /q "{p}"' if p.is_dir() else f'del /f /q "{p}"') for p in doomed)
        subprocess.Popen(f'cmd /d /c "ping -n 4 127.0.0.1 >nul & {cmds} & rmdir "{base}""',
                         creationflags=0x08000000 | 0x00000008, cwd=str(Path.home()))
    else:
        for p in doomed:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
        if not stay:
            base.rmdir()
    say("hearth is removed." + (f" Models and voices are kept in {base}." if keep_models else "")
        + " mantel's chats and settings are kept where they were.")


# ── command line (for the install scripts) ────────────────────────────────────

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
    try:
        found = conflicts()
        if found:
            print("hearth was not installed, so as not to disturb what is here:\n  - " + "\n  - ".join(found),
                  file=sys.stderr)
            return 3
        if a.cmd == "finish":
            finish(release(a.release, a.sha256), not a.no_mantel, not a.no_autostart, not a.no_start, not a.no_open,
                   lambda t: print(t, flush=True))
        return 0
    except Problem as e:
        print(e, file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"The connection failed ({e or type(e).__name__}). Run the installer again to go on.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
