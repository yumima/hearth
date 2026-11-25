"""hearth CLI — start, models, pull, bind, swap, roles, hardware.

`hearth start` is the one-shot: it brings up the bundled Ollama (if not
already running), then serves the OpenAI-compatible gateway on loopback.
"""

from __future__ import annotations

import argparse
import contextlib
import atexit
import errno
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import httpx

from . import __version__

from . import config as cfgmod
from . import catalog, hardware, ollama_supervisor, paths, procs
from .background import gateway_identity as _gateway_identity
from .term import ask


def _admin_base(cfg: cfgmod.Config) -> str:
    return f"http://{cfg.bind_host}:{cfg.bind_port}"


_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _pid_path() -> Path:
    """PID file for the running gateway, alongside config.yaml (~/.hearth)."""
    return cfgmod.config_path().parent / "hearth.pid"


def _pid_looks_like_hearth(pid: int) -> bool:
    """Is `pid` a hearth gateway? Asked before ending it: a pidfile left by a
    killed one may name another program's process by now."""
    return procs.runs(procs.argv(pid), "hearth", "serve")


def _listen_socket(host: str, port: int) -> socket.socket:
    """Claim the gateway's port before we announce anything.

    uvicorn binds late — after the lifespan has started — so leaving the bind
    to it means a port clash surfaces only once we have printed "serving ..."
    and written a pidfile, and the doomed process then deletes the live
    gateway's pidfile on its way out. Binding here turns a clash into an
    OSError we can report before touching any shared state. The socket is
    non-inheritable (Python's default), so a child we spawn later can't keep
    the port held after we exit.
    """
    # Same family rule uvicorn uses: IPv6 only for a literal IPv6 host. Letting
    # getaddrinfo pick could turn "localhost" into an IPv6-only ::1 listener
    # that clients dialing 127.0.0.1 can't reach.
    family = socket.AF_INET6 if ":" in (host or "") else socket.AF_INET
    infos = socket.getaddrinfo(host or None, port, family=family, type=socket.SOCK_STREAM,
                               flags=socket.AI_PASSIVE)
    family, socktype, proto, _canon, addr = infos[0]
    sock = socket.socket(family, socktype, proto)
    try:
        if sys.platform == "win32":
            # There SO_REUSEADDR lets two programs listen on one port at once.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(addr)
        # Listen now, not when uvicorn gets to it: with SO_REUSEADDR a bound
        # but non-listening port can still be bound by a racing second start.
        sock.listen(2048)
    except OSError:
        sock.close()
        raise
    return sock


def cmd_start(args: argparse.Namespace) -> int:
    """Start the engine so that it outlives whoever asked: the service where one is
    installed, else a process of its own. Returns when it answers."""
    from . import background

    cfg = cfgmod.load()
    if args.bind:
        host, _, port = args.bind.rpartition(":")
        cfg.bind_host, cfg.bind_port = host or cfg.bind_host, int(port)
    extra = []
    if args.bind:
        extra += ["--bind", args.bind]
    if args.no_manage:
        extra.append("--no-manage")
    if args.log_level != "info":
        extra += ["--log-level", args.log_level]
    service = None
    if not extra and _has_service():
        # Type=simple: started means forked, so what became of it is asked again.
        service = background.Service(lambda: _systemctl("start", _UNIT_NAME),
                                     lambda: _systemctl("is-active", "--quiet", _UNIT_NAME) == 0
                                     or _unit_is("activating"))
    elif cfg.bind_host not in _LOOPBACK:
        print("an address others can reach is for `hearth serve --unsafe-bind --api-key …`", file=sys.stderr)
        return 2
    # This interpreter, as it was called: not _hearth_exec_argv(), which keeps
    # the launcher link — starting the engine must not change what is installed.
    # -P: the folder it runs in is no place to import from.
    ok, message = background.start(cfg, [sys.executable, "-P", "-m", "hearth", "serve", *extra], service)
    print(f"[hearth] {message}", file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


def _has_service() -> bool:
    """Is the engine a service's here? (The service serves its own config at its own
    address: one named in the environment is an engine of its own.)"""
    own = any(os.environ.get(n) for n in ("HEARTH_HOME", "HEARTH_CONFIG", "HEARTH_DATA"))
    return not own and _unit_path().exists() and bool(shutil.which("systemctl"))


def _unit_runs() -> bool:
    return any(_unit_is(x) for x in ("active", "activating", "reloading", "deactivating"))


def _unit_is(state: str) -> bool:
    r = subprocess.run([shutil.which("systemctl") or "systemctl", "--user", "show", "-p", "ActiveState", "--value",
                        _UNIT_NAME], env=_sc_env(), capture_output=True, text=True, check=False)
    return r.stdout.strip() == state


def cmd_serve(args: argparse.Namespace) -> int:
    """The engine itself, in the foreground: what a service runs."""
    cfg = cfgmod.load()
    if args.bind:
        host, _, port = args.bind.rpartition(":")
        cfg.bind_host, cfg.bind_port = host or cfg.bind_host, int(port)

    allow_any_host = bool(args.unsafe_bind)
    api_key = args.api_key
    if allow_any_host and not api_key:
        print("--unsafe-bind requires --api-key (off-loopback exposure)", file=sys.stderr)
        return 2
    if cfg.bind_host not in _LOOPBACK and not allow_any_host:
        print(f"{cfg.bind_host} can be reached by others: that takes --unsafe-bind and --api-key", file=sys.stderr)
        return 2

    import uvicorn

    from .app import create_app

    # (systemd sets INVOCATION_ID for what it runs.)
    if "INVOCATION_ID" not in os.environ and _unit_path().exists():
        print("note: hearth is installed as a service, and this is not it: `hearth start` starts that one "
              "(Ctrl-C ends this one)", file=sys.stderr)

    # Detect + suggest the best-fit model: if the bound primary_chat differs from
    # what fits this hardware best, say so. A model that overflows VRAM runs its
    # spillover on the CPU and starves the GPU (slow). Best-effort; never blocks.
    # A remote-bound role has no VRAM footprint on this box, so the fit advice
    # is meaningless for it (and reads as a nag to abandon a deliberate choice).
    try:
        hw = hardware.probe()
        rec = catalog.recommend_roles(hw, [cfg.roles["primary_chat"].model] if "primary_chat" in cfg.roles else []).get("primary_chat")
        cur = cfg.roles.get("primary_chat")
        if rec and cur and cur.model != rec and not cfg.role_is_remote("primary_chat"):
            print(f"tip: primary_chat = '{cur.model}', but '{rec}' fits your hardware "
                  f"better (full-GPU speed).\n     apply with:  hearth model install {rec}   (or: hearth setup)", file=sys.stderr)
    except Exception:
        pass

    # Refuse to start a second gateway on a port that's already serving — a
    # double-start would orphan the first one (overwritten pidfile) and
    # bind-fail. Done before anything with a side effect (Ollama, pidfile) so a
    # duplicate start is a single line and no more.
    if (ident := _gateway_identity(cfg)) is not None:
        print(f"[hearth] a gateway is already serving on {_admin_base(cfg)} ({ident}) "
              "— use `hearth stop` first to restart", file=sys.stderr)
        return 0

    # Claim the port now: whoever holds it, we find out here rather than from
    # uvicorn after we've already claimed to be serving.
    try:
        sock = _listen_socket(cfg.bind_host, cfg.bind_port)
    except OSError as exc:
        where = f"http://{cfg.bind_host}:{cfg.bind_port}"
        if exc.errno == errno.EADDRINUSE or (sys.platform == "win32" and exc.errno == errno.EACCES):
            pid = procs.listening_on(cfg.bind_port)
            who = f" by pid {pid}" if pid else ""
            cl = procs.command_line(pid)[:90] if pid else ""
            print(f"[hearth] {where} is already in use{who} — not starting."
                  + (f"\n         holder: {cl}" if cl else "")
                  + "\n         stop the gateway that holds it (`hearth stop`), or serve elsewhere "
                    "with `hearth serve --bind 127.0.0.1:PORT`", file=sys.stderr)
        else:
            print(f"[hearth] cannot bind {where} — {exc}", file=sys.stderr)
        return 1

    # Bring up Ollama unless told not to.
    supervisor = None
    ollama = cfg.backends.get("ollama")
    if ollama and not args.no_manage:
        supervisor = ollama_supervisor.Supervisor(ollama.base_url, ollama.models_dir)
        if supervisor.start():
            print(f"[ollama] started ({ollama.base_url})" if ollama_supervisor.is_up(ollama.base_url) else
                  f"[ollama] not up yet ({ollama.base_url}) — it is started again as it ends; "
                  f"log: {ollama_supervisor.log_path()}")
        elif ollama_supervisor.is_up(ollama.base_url):
            print(f"[ollama] already running ({ollama.base_url})")
            # Only a daemon hearth starts gets these, and the recommended contexts
            # assume them: without q8_0 the KV cache doubles and spills to the CPU.
            print("[ollama] note: if hearth didn't start this daemon, run it with "
                  "OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0", file=sys.stderr)
        else:
            print(
                f"[ollama] WARNING not running and no program found at {ollama.base_url}; "
                "chat/embeddings will be unavailable until it's up",
                file=sys.stderr,
            )

    # Write a PID file so `hearth stop` / `hearth status` can find this gateway.
    # Only remove it on exit if it still holds OUR pid (don't clobber a pidfile
    # a newer start may have written).
    pid_file = _pid_path()
    my_pid = os.getpid()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(my_pid))

    def _cleanup_pid():
        try:
            if pid_file.exists() and pid_file.read_text().strip() == str(my_pid):
                pid_file.unlink()
        except OSError:
            pass

    atexit.register(_cleanup_pid)

    # Hint to run `hearth setup` when the primary_chat model isn't here yet.
    # Skipped for a remote-bound role: the provider hosts the weights, so there
    # is nothing to pull and downloading gemini-3.7-flash would just fail.
    pm = cfg.roles.get("primary_chat")
    if ollama and pm and not cfg.role_is_remote("primary_chat"):
        try:
            tags = httpx.get(f"{ollama.base_url}/api/tags", timeout=2.0).json().get("models", [])
            names = {t.get("name", "") for t in tags}
            if pm.model not in names:
                print(f"[hearth] primary_chat model '{pm.model}' not pulled — run "
                      f"`hearth setup` to auto-pick + pull a model that fits your hardware "
                      f"(or `hearth model install {pm.model}`)", file=sys.stderr)
        except httpx.HTTPError:
            pass

    app = create_app(cfg, allow_any_host=allow_any_host, api_key=api_key)
    print(f"[hearth] serving OpenAI-compatible API on http://{cfg.bind_host}:{cfg.bind_port}")
    print("[hearth] probe: GET /v1/models   health: GET /admin/health   chat: hearth chat")
    # server_header=False suppresses uvicorn's own "Server: uvicorn" so our
    # middleware's "Server: hearth/<v>" is the single, clean identity.
    # The socket is already bound (see _listen_socket) — uvicorn serves on it
    # instead of binding its own.
    config = uvicorn.Config(app, host=cfg.bind_host, port=cfg.bind_port,
                            log_level=args.log_level, server_header=False)
    server = uvicorn.Server(config)
    app.state.shutdown = lambda: setattr(server, "should_exit", True)
    if supervisor is not None:
        app.state.at_shutdown.append(supervisor.stop_watching)
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass  # uvicorn has shut down in order, and raises what it caught
    return 0 if server.started else 1


def cmd_model_list(args: argparse.Namespace) -> int:
    """What is on this computer and what uses it; with --all, what the providers have, too."""
    cfg = cfgmod.load()
    ollama = _ensure_ollama(cfg)
    here = {}
    if ollama is not None:
        try:
            here = {m["name"]: m.get("size", 0) for m in
                    httpx.get(f"{ollama.base_url}/api/tags", timeout=5.0).json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            pass
    from .provision import same_model

    used: dict[tuple[str, str], list[str]] = {}
    for role, r in cfg.roles.items():
        name = r.model
        if ollama is not None and r.backend == ollama.name:
            name = next((n for n in here if same_model(n, r.model)), r.model)
        used.setdefault((r.backend, name), []).append(role)
    for name, size in here.items():
        roles = used.pop((ollama.name, name), [])
        mark = f"  ⚠ {ollama.name}" if ollama.is_remote else ""
        print(f"{name:52s} {size / 1e9:6.1f} GB  {', '.join(roles) or '—'}{mark}")
    far = ollama is not None and ollama.is_remote
    for (backend, name), roles in used.items():
        remote = cfg.role_is_remote(roles[0])
        far |= remote
        where = ("⚠ " if remote else "") + backend if ollama is None or backend != ollama.name else "not here"
        print(f"{name:52s} {where:>9s}  {', '.join(roles)}")
    if far:
        print("\n⚠ a provider's model: what is sent to it leaves this computer.")
    if ollama is None:
        print("(Ollama is not running: what is on this computer is not known)", file=sys.stderr)
    if args.all:
        try:
            data = httpx.get(f"{_admin_base(cfg)}/v1/models", timeout=10.0).json().get("data", [])
        except (httpx.HTTPError, ValueError):
            print("the engine is not running: `hearth start`", file=sys.stderr)
            return 1
        print("\neverything that can be used:")
        for m in data:
            if m.get("owned_by") != "hearth-role":
                print(f"  {m['id']:50s} {m.get('owned_by', '')}")
    return 0


def _pull_model(ollama_base_url: str, model: str) -> int:
    """Stream an Ollama pull with a progress line. Returns 0 on success."""
    print(f"pulling {model} ...")
    last = ""
    try:
        with httpx.stream(
            "POST", f"{ollama_base_url}/api/pull",
            json={"model": model, "stream": True}, timeout=None,
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                status = obj.get("status", "")
                total, completed = obj.get("total"), obj.get("completed")
                if total and completed:
                    msg = f"\r{status} {100.0*completed/total:5.1f}% ({completed/1e9:.2f}/{total/1e9:.2f} GB)"
                else:
                    msg = f"\r{status}"
                if msg != last:
                    sys.stdout.write(msg.ljust(70))
                    sys.stdout.flush()
                    last = msg
                if obj.get("error"):
                    print(f"\nerror: {obj['error']}", file=sys.stderr)
                    print(_ollama_upgrade_hint(ollama_base_url), file=sys.stderr)
                    return 1
            if r.status_code >= 400:
                r.read()
                detail = _pull_error(r)
                print(f"\npull failed: {detail}", file=sys.stderr)
                print(_ollama_upgrade_hint(ollama_base_url), file=sys.stderr)
                return 1
    except httpx.HTTPError as e:
        print(f"\npull failed: {e}", file=sys.stderr)
        return 1
    print("\ndone")
    return 0


def _pull_error(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error") or f"HTTP {r.status_code}")
    except ValueError:
        return r.text.strip() or f"HTTP {r.status_code}"


def _ollama_upgrade_hint(ollama_base_url: str) -> str:
    """A new model refused by an old daemon otherwise reads as a dead end."""
    try:
        version = httpx.get(f"{ollama_base_url}/api/version", timeout=3.0).json().get("version")
    except (httpx.HTTPError, ValueError):
        version = None
    have = f"Ollama {version}" if version else "your Ollama"
    return (f"hint: {have} may be too old for this model. Update it "
            "(https://github.com/ollama/ollama/releases, into vendor/ollama) and retry.")


def _ensure_ollama(cfg: cfgmod.Config):
    """Return the ollama backend with its daemon up, or None."""
    ollama = cfg.backends.get("ollama")
    if not ollama:
        return None
    if not ollama_supervisor.is_up(ollama.base_url):
        if _gateway_identity(cfg) is not None:
            # Its to start: it is doing that, or has said in the log why it cannot.
            print("waiting for the running gateway's Ollama…")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not ollama_supervisor.is_up(ollama.base_url):
                time.sleep(0.5)
        else:
            ollama_supervisor.Supervisor(ollama.base_url, ollama.models_dir).start(keep_up=False)
        if not ollama_supervisor.is_up(ollama.base_url):
            return None
    return ollama


# ── local TTS voice (Piper) ──────────────────────────────────────────────────
#
# Speech is part of the local AI runtime, so `hearth setup`/`hearth voice install`
# provision it: install the piper CLI (into hearth's venv, symlinked onto PATH)
# and download a voice model to hearth's own folder (paths.voices_dir), where finterm's
# read-aloud auto-discovers — no manual config.

_DEFAULT_VOICE = "en_US-amy-medium"
_KOKORO = "kokoro"  # the natural voices, all at once


def _voice_dir() -> Path:
    return paths.voices_dir("piper")


def _voice_url(voice: str, ext: str) -> str:
    # piper voice id "en_US-amy-medium" → HF path "en/en_US/amy/medium".
    parts = voice.split("-", 2)
    if len(parts) != 3:
        raise ValueError(f"voice id should look like 'en_US-amy-medium', got {voice!r}")
    region, name, quality = parts
    lang = region.split("_")[0]
    return ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang}/{region}/{name}/{quality}/{voice}{ext}")


def _download(url: str, dest: Path) -> bool:
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as r:
            r.raise_for_status()
            expected = int(r.headers.get("content-length") or 0)
            written = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(65536):
                    f.write(chunk)
                    written += len(chunk)
        if written == 0 or (expected and written < expected):
            dest.unlink(missing_ok=True)
            print(f"  download incomplete ({written}/{expected or '?'} bytes)", file=sys.stderr)
            return False
        return True
    except (httpx.HTTPError, OSError) as e:
        dest.unlink(missing_ok=True)
        print(f"  download failed: {e}", file=sys.stderr)
        return False


_PIPER_RELEASE = (
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
)


def _ensure_piper() -> bool:
    """Make the `piper` CLI available on PATH. Uses piper's self-contained native
    binary (bundles its own onnxruntime + espeak-ng-data) rather than the
    `piper-tts` pip package — the pip path has no wheels for very new Pythons
    (e.g. 3.14). Extracts to ~/.local/share/piper and symlinks onto ~/.local/bin."""
    if shutil.which("piper") or shutil.which("piper-tts"):
        return True
    if not (sys.platform.startswith("linux") and platform.machine() == "x86_64"):
        print("  Piper voices can be installed on 64-bit Linux only, for now; the natural voices "
              "(hearth voice install kokoro) work here", file=sys.stderr)
        return False
    pdir = Path.home() / ".local" / "share" / "piper"
    bin_ = pdir / "piper" / "piper"  # the release tars to a `piper/` dir
    if not bin_.exists():
        pdir.mkdir(parents=True, exist_ok=True)
        tar = pdir / "piper.tar.gz"
        print("  downloading the piper runtime (native binary)…")
        if not _download(_PIPER_RELEASE, tar):
            return False
        import tarfile
        try:
            with tarfile.open(tar) as t:
                t.extractall(pdir, filter="data")  # filter required on 3.14
        except (tarfile.TarError, OSError, TypeError) as e:
            print(f"  extracting piper failed: {e}", file=sys.stderr)
            return False
        tar.unlink(missing_ok=True)
    if not bin_.exists():
        print("  piper binary missing after extract", file=sys.stderr)
        return False
    bin_.chmod(0o755)
    link = Path.home() / ".local" / "bin" / "piper"
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(bin_)
    except OSError as e:
        print(f"  could not symlink piper onto PATH: {e}", file=sys.stderr)
        return False
    print(f"  ✓ piper → {link}")
    return True


def _provision_voice(voice: str) -> bool:
    """Provision a local Piper voice (CLI + model) that finterm auto-discovers.
    Best-effort: never raises (returns False), and removes a half-downloaded
    voice so finterm can't discover a broken one."""
    try:
        if not _ensure_piper():
            return False
        vdir = _voice_dir()
        vdir.mkdir(parents=True, exist_ok=True)
        for ext in (".onnx", ".onnx.json"):
            dest = vdir / (voice + ext)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            print(f"  downloading {voice}{ext}…")
            if not _download(_voice_url(voice, ext), dest):
                for e2 in (".onnx", ".onnx.json"):  # don't leave a .onnx without its .json
                    (vdir / (voice + e2)).unlink(missing_ok=True)
                return False
        print(f"  ✓ voice ready: {vdir / (voice + '.onnx')}")
        return True
    except Exception as e:  # voice provisioning must never crash setup
        print(f"  voice provisioning error: {e}", file=sys.stderr)
        return False


def cmd_voice_install(args: argparse.Namespace) -> int:
    if args.name == _KOKORO:
        return _run_job("kokoro")
    if _provision_voice(args.name):
        return 0
    print("the voice could not be installed.", file=sys.stderr)
    return 1


def cmd_voice_list(args: argparse.Namespace) -> int:
    from . import kokoro_tts

    print(f"{_KOKORO:32s} {'installed' if kokoro_tts.installed() else '—'}   the natural voices")
    for d in paths.voice_dirs("piper"):
        for p in sorted(d.glob("*.onnx")):
            print(f"{p.stem:32s} installed   Piper")
    return 0


def cmd_voice_remove(args: argparse.Namespace) -> int:
    engine, name = (_KOKORO, "") if args.name == _KOKORO else ("piper", args.name)
    if engine != _KOKORO and (not name or Path(name).name != name or name.startswith(".")):
        print(f"not a voice: {name!r}", file=sys.stderr)
        return 2
    d = paths.voices_dir(engine)
    doomed = [p for p in ([d] if engine == _KOKORO else [d / f"{name}.onnx", d / f"{name}.onnx.json"]) if p.exists()]
    if not doomed:
        elsewhere = [x for x in paths.voice_dirs(engine) if x != d and (not name or (x / f"{name}.onnx").exists())]
        print(f"{args.name} is not in {d}" + (f"; it is in {elsewhere[0]}, which is not hearth's own" if elsewhere else ""),
              file=sys.stderr)
        return 1
    for p in doomed:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    print(f"removed {args.name}")
    return 0


# ── models and voices: the same jobs as on the setup page ────────────────────


class _Jobs:
    """The running gateway's jobs, or ones of our own when there is none."""

    def __init__(self, cfg: cfgmod.Config):
        self.cfg = cfg
        self.base = _admin_base(cfg) if _gateway_identity(cfg) else None
        self.own = None
        if self.base is None:
            from .provision import Provisioner

            if _ensure_ollama(cfg) is None:
                raise RuntimeError("Ollama is not available: " + (ollama_supervisor.last_lines(3) or "no program found"))
            self.own = Provisioner(cfg)

    def _ask(self, method: str, path: str, **kw) -> dict:
        try:
            r = httpx.request(method, self.base + path, timeout=30.0, **kw)
            data = r.json() if r.content else {}
        except (httpx.HTTPError, ValueError) as e:
            raise RuntimeError(f"hearth stopped answering ({e or type(e).__name__}). Start it, and run this again to go on.") from e
        if r.status_code >= 400:
            raise RuntimeError(data.get("error") or f"HTTP {r.status_code}")
        return data

    def state(self) -> dict:
        return self.own.state() if self.own else self._ask("GET", "/admin/setup")

    def remove(self, model: str) -> None:
        if self.own:
            self.own.remove(model)
        else:
            self._ask("DELETE", f"/admin/models/{model}")

    def run(self, kind: str, what: str = "", anyway: bool = False) -> dict:
        """Submit, show its progress, return it finished. Ctrl-C stops it."""
        body = {"kind": kind, "what": what, "anyway": anyway}
        job = self.own.submit(**body).as_dict() if self.own else self._ask("POST", "/admin/jobs", json=body)
        shown = ""
        try:
            while job["stage"] not in ("done", "failed", "stopped"):
                text = job["message"] or "Waiting…"
                if job["stage"] == "download" and job["total"]:
                    text += f"  {job['done'] / 1e9:.2f} of {job['total'] / 1e9:.2f} GB"
                if text != shown:
                    print("\r" + text.ljust(78)[:78], end="", flush=True)
                    shown = text
                time.sleep(0.5)
                jobs = self.state()["jobs"] if self.base else [j.as_dict() for j in self.own._jobs]
                found = next((j for j in jobs if j["id"] == job["id"] and j["what"] == what), None)
                if found is None:
                    raise RuntimeError("hearth was restarted meanwhile. Run this again to go on.")
                job = found
        except KeyboardInterrupt:
            if self.own:
                self.own.stop_job(job["id"])
            else:
                self._ask("POST", f"/admin/jobs/{job['id']}/stop")
            job = {**job, "stage": "stopped", "message": "Stopped. Start it again to go on."}
        print("\r" + " " * 78 + "\r" + job["message"])
        return job


def _jobs() -> _Jobs | None:
    try:
        return _Jobs(cfgmod.load())
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return None


def _run_job(kind: str, what: str = "", anyway: bool = False) -> int:
    from .provision import Problem

    jobs = _jobs()
    if jobs is None:
        return 1
    try:
        job = jobs.run(kind, what, anyway)
    except (RuntimeError, Problem) as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    if job.get("can_anyway"):
        print(f"to use it all the same:  hearth model install {what} --anyway", file=sys.stderr)
    return 0 if job["stage"] == "done" else 1


def cmd_model_suggest(args: argparse.Namespace) -> int:
    plan = catalog.plan(hardware.probe())
    if args.json:
        print(json.dumps(plan, indent=2))
        return 0
    print(plan["summary"])
    for o in plan["offers"]:
        mark = "★" if o["recommended"] else " "
        print(f" {mark} {o['title']:24s} {o['download'] / 1e9:5.1f} GB  {o['note']}\n     {o['model']}")
    return 0


def cmd_model_install(args: argparse.Namespace) -> int:
    if not args.download_only:
        return _run_job("model", args.model, args.anyway)
    ollama = _ensure_ollama(cfgmod.load())
    if ollama is None:
        print("Ollama is not available", file=sys.stderr)
        return 1
    return _pull_model(ollama.base_url, args.model)


def cmd_model_remove(args: argparse.Namespace) -> int:
    from .provision import Problem

    jobs = _jobs()
    if jobs is None:
        return 1
    try:
        jobs.remove(args.model)
    except (RuntimeError, Problem, httpx.HTTPError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"removed {args.model}: its files are deleted")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """What the setup page does, here: the model that suits this computer, then
    the natural voices."""
    plan = catalog.plan(hardware.probe())
    print(plan["summary"])
    best = next((o for o in plan["offers"] if o["recommended"]), None)
    if best is None:
        print("No model is recommended for this computer. `hearth model suggest` lists them all; a cloud model can be "
              "used with `hearth provider add`.")
        return 1
    print(f"Recommended: {best['title']} — {best['download'] / 1e9:.1f} GB, {best['license']}. {best['note']}")
    if not args.yes:
        try:
            if input("Install it? [Y/n] ").strip().lower() in ("n", "no"):
                return 0
        except EOFError:
            return 1
    if (rc := _run_job("model", best["model"])) != 0:
        return rc
    if not args.no_voice:
        _run_job("kokoro")
    print("Done. Try: hearth")
    return 0


def cmd_model_use(args: argparse.Namespace) -> int:
    """Which model answers for what. A model of this computer, for the chat: as it
    is installed, measured and checked. Anything else is as it is said."""
    from .provision import CHAT_ROLES

    cfg = cfgmod.load()
    ollama = next((n for n, b in cfg.backends.items() if b.type == "ollama"), "ollama")
    provider = args.provider or ollama
    if provider not in cfg.backends:
        print(f"no provider {provider!r}: `hearth provider list`", file=sys.stderr)
        return 2
    if args.context is not None and args.context < 0:
        print("--context must be >= 0", file=sys.stderr)
        return 2
    if not args.roles and provider == ollama and args.context is None:
        return _run_job("model", args.model, args.anyway)
    return max(_bind(cfg, role, args.model, provider, args.context) for role in (args.roles or CHAT_ROLES))


def _bind(cfg: cfgmod.Config, role: str, model: str, backend: str, ctx_arg: int | None) -> int:
    # A role's context is tuned to the MODEL that was bound (what fits in VRAM
    # at full-GPU speed), so it can't carry over to a different one. Dropping it
    # silently is the trap: the daemon falls back to its own default and starts
    # discarding the oldest messages — which reads as the model ignoring its
    # system prompt, not as a config change. So say it out loud.
    previous = cfg.roles.get(role)
    prior_ctx = int(previous.context or 0) if previous else 0
    # With no --context, match a role already on this model: Ollama reloads a
    # model whose context changes, so hopping between the two would cost a cold
    # start every time. An explicit --context 0 still clears it.
    shared_ctx = cfg.context_of_model(model, exclude_role=role) if ctx_arg is None else 0
    if shared_ctx:
        ctx_arg = shared_ctx
    cfg.roles[role] = cfgmod.RoleBinding(
        model=model, backend=backend, context=int(ctx_arg or 0))
    cfgmod.save(cfg)
    print(f"{role} -> {model} ({backend})")
    if shared_ctx:
        print(f"context: {shared_ctx} (matches other roles on this model)")
    elif prior_ctx and not ctx_arg:
        print(f"note: cleared this role's context ({prior_ctx}); the daemon default "
              f"applies now.\n      keep it with:  hearth model use {model} --for {role} "
              f"--provider {backend} --context {prior_ctx}")
    elif ctx_arg:
        print(f"context: {int(ctx_arg)}")
    # If a gateway is running, hot-apply via admin so no restart is needed.
    base = _admin_base(cfg)
    try:
        r = httpx.put(
            f"{base}/admin/roles/{role}",
            json={"model": model, "backend": backend,
                  "context": int(ctx_arg or 0)}, timeout=3.0,
        )
    except httpx.HTTPError:
        return 0  # no gateway running; the saved config applies when one starts
    if r.is_success:
        print("(applied to running gateway)")
    else:
        print(f"warning: saved, but the running gateway refused it (HTTP {r.status_code}) "
              "and keeps its old binding; restart it to apply.", file=sys.stderr)
    return 0


def cmd_provider(args: argparse.Namespace) -> int:
    """Where models are: the local Ollama, or a remote
    OpenAI-compatible API (cloud GLM/OpenAI/Groq/… or a self-hosted vLLM)."""
    cfg = cfgmod.load()
    action = getattr(args, "action", None)

    if action in (None, "list"):
        if not cfg.backends:
            print("(no providers)")
        for name, b in cfg.backends.items():
            key = "literal" if b.api_key else (f"env:{b.api_key_env}" if b.api_key_env else "—")
            print(f"{name:12s} type={b.type:7s} base_url={b.base_url}  api_key={key}")
        return 0

    if action == "add":
        if args.type == "openai" and not (args.api_key or args.api_key_env):
            print("an openai provider needs --api-key or --api-key-env", file=sys.stderr)
            return 2
        cfg.backends[args.name] = cfgmod.Backend(
            args.name, args.type, args.base_url, args.api_key or "", args.api_key_env or "")
        cfgmod.save(cfg)
        print(f"added provider {args.name!r} ({args.type}) -> {args.base_url}")
        print("next: the engine loads it when it is restarted  →  hearth restart")
        print(f"      then use a model of it                    →  hearth model use <model> --provider {args.name}")
        return 0

    if action == "remove":
        b = cfg.backends.get(args.name)
        if b is None:
            print(f"no such provider {args.name!r}", file=sys.stderr)
            return 2
        if b.type == "ollama":
            print("the local Ollama stays", file=sys.stderr)
            return 2
        bound = [rn for rn, rb in cfg.roles.items() if rb.backend == args.name]
        if bound:
            print(f"note: choose another model for: {', '.join(bound)}", file=sys.stderr)
        del cfg.backends[args.name]
        cfgmod.save(cfg)
        print(f"removed provider {args.name!r}; `hearth restart` to apply.")
        return 0
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run the engine's web search directly — no model in the loop.

    The point is separating "search is broken" from "the model didn't search":
    this exercises the exact provider chain /v1 uses and names which provider
    answered, so a bad key or a changed markup shows up here first.
    """
    import asyncio

    from . import websearch

    bold, dim, reset = "\033[1m", "\033[2m", "\033[0m"
    cfg = cfgmod.load()
    scfg = cfg.search
    if args.provider:
        scfg.provider = args.provider

    if args.fetch:
        out = asyncio.run(websearch.fetch(args.query, scfg))
        if "error" in out:
            print(out["error"], file=sys.stderr)
            return 1
        print(f"{dim}[{out['status']}] {out['url']} ({out['content_type']}){reset}\n")
        print(out["text"] + ("\n…(truncated)" if out["truncated"] else ""))
        return 0

    try:
        results = asyncio.run(websearch.search(args.query, scfg, args.count))
    except websearch.SearchError as e:
        print(f"search failed — {e}", file=sys.stderr)
        return 1
    print(f"{dim}provider: {results[0].source}{reset}\n")
    for i, r in enumerate(results, 1):
        age = f" {dim}({r.age}){reset}" if r.age else ""
        print(f"{bold}{i}. {r.title}{reset}{age}\n   {dim}{r.url}{reset}")
        if r.snippet:
            print(f"   {r.snippet}")
        print()
    return 0


_CODE_HELP = """\
{dim}commands:
  /add <file>     pin a file into the conversation context
  /model [name]   show/switch the model or role (default: coding)
  /auto           toggle auto-approve for edits, commands and the web (off by default)
  /chat           talk without files (/code comes back here)
  /clear          forget the conversation (keep the project context)
  /help  /exit    (Ctrl-C ends the task, not the session)
just type a request — the agent greps, reads, edits, runs shell and searches
the web to do it. non-interactive:  hearth -p "fix the failing test"{reset}"""

_SWITCH = -1  # from one interactive mode to the other


def _modes(mode: str, args: argparse.Namespace) -> int:
    """The terminal's two modes: the coding agent (plain `hearth`), and the chat.
    Each keeps its conversation while the other is used."""
    try:
        import readline  # noqa: F401  (line editing and history; not on Windows)
    except ImportError:
        pass
    args.models = {mode: args.model}  # the one asked for is for the mode it was started in
    other = {"chat": "code", "code": "chat"}
    came_from = None
    while True:
        args.model = args.models.get(mode)
        rc = (_chat_mode if mode == "chat" else _code_mode)(args)
        if rc == _SWITCH:
            mode, came_from = other[mode], mode
        elif rc != 0 and came_from is not None:  # it could not be gone to: the one before goes on
            mode, came_from = came_from, None
        else:
            return rc


def _engine_ready(cfg: cfgmod.Config, role: str, model: str | None) -> str | None:
    """The engine's address, started if it was not. None, and said, if there is no
    engine or no model to talk to."""
    from . import background

    base = _admin_base(cfg)
    if background.gateway_identity(cfg) is None:
        print("Starting hearth…", file=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):  # what is asked for may be read by a program
            if cmd_start(build_parser().parse_args(["start"])) != 0:
                return None
    try:
        roles = httpx.get(f"{base}/admin/roles", timeout=10.0).json().get("roles", {})
    except (httpx.HTTPError, ValueError) as e:
        print(f"hearth does not answer at {base}: {str(e) or type(e).__name__}", file=sys.stderr)
        return None
    if model is None and role not in roles:
        print(f"No model is chosen yet. Choose one at {base}/ , or run `hearth setup`.", file=sys.stderr)
        return None
    return base


def _trusted_path() -> Path:
    return cfgmod.config_path().parent / "trusted.json"


def _trusts(root: Path) -> bool:
    """Whether the person said so, of this folder or one it is in; asked if not."""
    try:
        known = [str(p) for p in json.loads(_trusted_path().read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError):
        known = []
    if any(root == Path(k) or Path(k) in root.parents for k in known):
        return True
    wide = root == Path.home() or root in Path.home().parents
    print(f"hearth will work in {root}" + (
        "\nThat is everything you have: better to start it in a project's folder." if wide else ""))
    print("It can read what is in there. It asks before it changes a file, runs a command or goes to the web.")
    try:
        if input("Do you trust what is in this folder? [y/N] ").strip().lower() not in ("y", "yes"):
            return False
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not wide:  # home is asked for each time, and does not answer for what is in it
        p = _trusted_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps([*known, str(root)], indent=1), encoding="utf-8")
        tmp.replace(p)
    return True


def cmd_code(args: argparse.Namespace) -> int:
    """Coding agent: the model reads and edits files + runs shell in a folder."""
    return _modes("code", args)


def _code_mode(args: argparse.Namespace) -> int:
    root = Path(getattr(args, "path", None) or ".").resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    prompt = getattr(args, "prompt", None)
    if not prompt and not (sys.stdin and sys.stdin.isatty()):
        print('hearth asks before it changes something, and there is nobody here to ask. '
              'For one task without asking: hearth -p "…"', file=sys.stderr)
        return 2
    if not prompt and not hasattr(args, "agent") and not _trusts(root):
        return 1
    if hasattr(args, "agent"):
        cfg = cfgmod.load()  # come back to: what was said stays, whatever became of the engine
        base = _engine_ready(cfg, "coding", args.agent.model) or _admin_base(cfg)
    else:
        base = _engine_ready(cfgmod.load(), "coding", args.model)
    if base is None:
        return 1

    from .code_agent import CodeAgent
    model = args.model or "coding"
    # Non-interactive mode: one task, then exit. This is what makes the agent
    # scriptable — usable from a Makefile, a git hook, or another agent — and
    # it implies auto-approve because there is nobody at the terminal to ask.
    if prompt:
        agent = CodeAgent(str(root), base, model, auto=True,
                          max_rounds=args.max_rounds, quiet=bool(args.quiet))
        out = agent.ask(prompt)
        if args.quiet and out:
            print(out)
        return 0 if out else 1

    agent = getattr(args, "agent", None)
    if agent is None:
        agent = args.agent = CodeAgent(str(root), base, model, auto=bool(getattr(args, "yolo", False)),
                                       max_rounds=getattr(args, "max_rounds", 60))

    bold, dim, reset = "\033[1m", "\033[2m", "\033[0m"
    print(f"{dim}hearth — {agent.root} · model {agent.model}"
          f"{' · AUTO-APPROVE' if agent.auto else ''} · /help · /exit{reset}")
    while True:
        try:
            user = ask(f"\n{bold}hearth>{reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/chat":
            args.models["code"] = agent.model
            return _SWITCH
        if user in ("/exit", "/quit"):
            break
        if user in ("/help", "/?"):
            print(_CODE_HELP.format(dim=dim, reset=reset))
            continue
        if user == "/clear":
            agent.messages = agent.messages[:1]
            print(f"{dim}(context cleared){reset}")
            continue
        if user == "/auto":
            agent.auto = not agent.auto
            print(f"{dim}(auto-approve {'ON' if agent.auto else 'OFF'}){reset}")
            continue
        if user == "/model" or user.startswith("/model "):
            arg = user[len("/model"):].strip()
            if arg:
                agent.model = arg
                print(f"{dim}(model → {arg}){reset}")
            else:
                concrete, roles = _chat_models(base)
                print(f"{dim}models: {'  '.join(concrete)}\nroles:  {'  '.join(roles)} "
                      f"(current: {agent.model}){reset}")
            continue
        if user == "/add" or user.startswith("/add "):
            arg = user[len("/add"):].strip()
            try:
                p = agent._safe(arg)
                agent.messages.append({"role": "user",
                    "content": f"For reference, here is {arg}:\n\n{p.read_text(errors='replace')[:60000]}"})
                print(f"{dim}(added {arg} to context){reset}")
            except (OSError, ValueError) as e:
                print(f"{dim}(add failed: {e}){reset}")
            continue
        if user.startswith("/"):
            print(f"{dim}(unknown command {user.split()[0]!r} — /help){reset}")
            continue
        agent.ask(user)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    base = _admin_base(cfg)
    try:
        ver = httpx.get(f"{base}/admin/version", timeout=3.0).json()
        health = httpx.get(f"{base}/admin/health", timeout=3.0).json()
    except httpx.HTTPError:
        print(f"hearth: NOT running at {base}  (start with `hearth start`)")
        print(f"at login: {_at_login()}")
        return 1
    print(f"hearth {ver.get('version')}  (contract {ver.get('contract')})  @ {base}")
    print(f"status: {health.get('status')}   providers: {health.get('backends')}   at login: {_at_login()}")
    try:
        roles = httpx.get(f"{base}/admin/roles", timeout=3.0).json().get("roles", {})
        remote_any = False
        for k, v in roles.items():
            remote = bool(v.get("remote"))
            remote_any = remote_any or remote
            print(f"  {k:14s} -> {v['model']}{'  ⚠ off-box' if remote else ''}")
        if remote_any:
            print("\n⚠ marked roles send prompts to a remote provider — "
                  "they leave this machine.")
    except httpx.HTTPError:
        pass
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    if _has_service() and _unit_runs():
        # systemd's to end: what it keeps up, it would bring back.
        rc = _systemctl("stop", _UNIT_NAME)
        print("stopped the hearth service" if rc == 0 else f"the hearth service did not stop (exit {rc})",
              file=sys.stdout if rc == 0 else sys.stderr)
        if rc != 0 or _engine_pid()[0] is None:
            return rc
        # …and one that was started beside it
    pid, via = _engine_pid()
    if pid is None:
        print(f"no running hearth gateway found "
              f"(no pidfile; nothing hearth-like listening on :{cfgmod.load().bind_port})",
              file=sys.stderr)
        return 1
    if not _ask_to_stop(cfgmod.load(), pid):
        procs.end_tree(pid)
    if procs.alive(pid):
        print(f"could not stop pid {pid} (another user's?)", file=sys.stderr)
        return 1
    print(f"stopped hearth gateway (pid {pid}, via {via})")
    _pid_path().unlink(missing_ok=True)
    return 0


def _engine_pid() -> tuple[int | None, str]:
    """The engine's process, and how it was found."""
    pid_file = _pid_path()
    pid: int | None = None
    via = "pidfile"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)
            print("invalid pidfile (removed)", file=sys.stderr)
            pid = None
        # Guard against a stale pidfile whose PID was reused by another process.
        if pid is not None and not _pid_looks_like_hearth(pid):
            pid_file.unlink(missing_ok=True)
            print(f"stale pidfile: pid {pid} is not a hearth gateway (removed, did not signal)",
                  file=sys.stderr)
            pid = None
    if pid is None:
        # No (valid) pidfile — the gateway may have been started outside this
        # CLI (e.g. `python -m hearth serve`). Find it by the port it serves.
        port = cfgmod.load().bind_port
        found = procs.listening_on(port)
        if found is not None and _pid_looks_like_hearth(found):
            pid, via = found, f"port {port}"
    return pid, via


def _ask_to_stop(cfg: cfgmod.Config, pid: int, wait_s: float = 15.0) -> bool:
    """Have the gateway end itself, its Ollama with it. False: it did not."""
    if procs.listening_on(cfg.bind_port) not in (pid, None):
        return False  # another gateway has the configured port
    try:
        if httpx.post(f"{_admin_base(cfg)}/admin/shutdown", timeout=3.0).status_code != 200:
            return False
    except httpx.HTTPError:
        return False
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not procs.alive(pid):
            return True
        time.sleep(0.1)
    return False


_CHAT_HELP = """\
{dim}commands:
  /help  /?           show this help
  /model [name]       switch model — no name shows a numbered picker
  /models             list servable models + role aliases
  /think on|off       deep-think (off = fast lane: fast_chat)
  /show               toggle full reasoning vs the compact spinner
  /system [text]      show, or set, the system prompt
  /status             model, deep-think, gateway health, turn count
  /retry              regenerate the last reply
  /save [file]        save the transcript (default: hearth-chat.md)
  /code               the coding agent, in this folder (/chat comes back here)
  /clear              clear the screen and the conversation
  /reset              clear the conversation (keep the screen)
  /exit  /quit        leave{reset}"""


def _chat_models(base: str) -> tuple[list[str], list[str]]:
    """(concrete model ids, role aliases) from the gateway's /v1/models."""
    try:
        r = httpx.get(f"{base}/v1/models", timeout=5.0)
        r.raise_for_status()
        data = r.json().get("data", [])
    except (httpx.HTTPError, ValueError):
        return [], []
    concrete = [m["id"] for m in data if m.get("id") and m.get("owned_by") != "hearth-role"]
    roles = [m["id"] for m in data if m.get("id") and m.get("owned_by") == "hearth-role"]
    return concrete, roles


def cmd_chat(args: argparse.Namespace) -> int:
    return _modes("chat", args)


def _chat_mode(args: argparse.Namespace) -> int:
    """A tiny streaming REPL against the local gateway — like ChatGPT in the
    terminal, but fully local."""
    cfg = cfgmod.load()
    base = _engine_ready(cfg, "primary_chat", args.model)
    if base is None and hasattr(args, "history"):
        base = _admin_base(cfg)  # come back to: what was said stays, whatever became of the engine
    if base is None:
        return 1

    model = args.model or "primary_chat"
    think_on = not bool(getattr(args, "no_think", False))    # deep-think, default on
    show_think = bool(getattr(args, "show_thinking", False))  # show full reasoning text
    think_on, show_think = getattr(args, "chat_settings", (think_on, show_think))
    bold, dim, reset = "\033[1m", "\033[2m", "\033[0m"
    print(f"{dim}hearth chat — model {model} · deep-think {'on' if think_on else 'off'} · "
          f"/help for commands · /exit to quit{reset}")
    history: list[dict] = getattr(args, "history", [])
    if not history and getattr(args, "system", None):
        history.append({"role": "system", "content": args.system})

    while True:
        try:
            user = ask(f"\n{bold}you>{reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/code":
            args.history, args.models["chat"], args.chat_settings = history, model, (think_on, show_think)
            return _SWITCH
        if user in ("/exit", "/quit"):
            break
        if user in ("/help", "/?", "/commands"):
            print(_CHAT_HELP.format(dim=dim, reset=reset))
            continue
        if user == "/clear":
            history = [h for h in history if h["role"] == "system"]
            sys.stdout.write("\033[H\033[2J\033[3J")  # home + clear screen + scrollback
            sys.stdout.flush()
            print(f"{dim}(cleared){reset}")
            continue
        if user == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print(f"{dim}(conversation reset){reset}")
            continue
        if user == "/models":
            concrete, roles = _chat_models(base)
            if not concrete and not roles:
                print(f"{dim}(could not reach {base}){reset}")
            else:
                print(f"{dim}models:  {'  '.join(concrete)}{reset}")
                if roles:
                    print(f"{dim}roles:   {'  '.join(roles)}{reset}")
            continue
        if user == "/model" or user.startswith("/model "):
            arg = user[len("/model"):].strip()
            if arg:
                model = arg
                print(f"{dim}(model → {model}){reset}")
                continue
            concrete, roles = _chat_models(base)
            choices = concrete + roles
            if not choices:
                print(f"{dim}(could not list models at {base}){reset}")
                continue
            print(f"{dim}select a model:{reset}")
            for i, m in enumerate(choices, 1):
                tag = " (current)" if m == model else (" ·role" if m in roles else "")
                print(f"  {dim}{i:>2}{reset} {m}{dim}{tag}{reset}")
            try:
                sel = ask(f"{bold}model #/name>{reset} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                sel = ""
            if not sel:
                print(f"{dim}(unchanged){reset}")
            elif sel.isascii() and sel.isdigit() and 1 <= int(sel) <= len(choices):
                model = choices[int(sel) - 1]
                print(f"{dim}(model → {model}){reset}")
            else:
                model = sel
                print(f"{dim}(model → {model}){reset}")
            continue
        if user == "/think" or user.startswith("/think "):
            arg = user[len("/think"):].strip().lower()
            if arg not in ("", "on", "off"):
                print(f"{dim}(usage: /think on|off){reset}")
                continue
            think_on = {"on": True, "off": False, "": not think_on}[arg]
            if think_on:
                print(f"{dim}(deep-think ON — model {model}, reasoning enabled){reset}")
            else:
                print(f"{dim}(deep-think OFF — fast lane: fast_chat, no reasoning){reset}")
            continue
        if user == "/show":
            show_think = not show_think
            print(f"{dim}(reasoning display: {'full text' if show_think else 'compact indicator'}){reset}")
            continue
        if user == "/system" or user.startswith("/system "):
            arg = user[len("/system"):].strip()
            if not arg:
                cur = next((h["content"] for h in history if h["role"] == "system"), None)
                print(f"{dim}system: {cur if cur else '(none)'}{reset}")
            else:
                history = [h for h in history if h["role"] != "system"]
                history.insert(0, {"role": "system", "content": arg})
                print(f"{dim}(system prompt set){reset}")
            continue
        if user == "/status":
            n_turns = sum(1 for h in history if h["role"] == "user")
            try:
                up = httpx.get(f"{base}/admin/health", timeout=2.0).status_code in (200, 503)
            except httpx.HTTPError:
                up = False
            has_sys = any(h["role"] == "system" for h in history)
            print(f"{dim}model: {model} · deep-think: {'on' if think_on else 'off'} · "
                  f"reasoning: {'full' if show_think else 'compact'}\n"
                  f"gateway: {base} ({'up' if up else 'down'}) · turns: {n_turns} · "
                  f"system: {'set' if has_sys else 'none'}{reset}")
            continue
        if user == "/save" or user.startswith("/save "):
            path = user[len("/save"):].strip() or "hearth-chat.md"
            try:
                body = "\n".join(f"### {h['role']}\n\n{h['content']}\n" for h in history)
                Path(path).write_text(body, encoding="utf-8")
                print(f"{dim}(saved {len(history)} messages → {path}){reset}")
            except OSError as e:
                print(f"{dim}(save failed: {e}){reset}")
            continue
        if user == "/retry":
            # Drop the last assistant reply + re-send the last user message;
            # fall through (no `continue`) so the send block below re-runs it.
            if history and history[-1]["role"] == "assistant":
                history.pop()
            last_user = None
            for i in range(len(history) - 1, -1, -1):
                if history[i]["role"] == "user":
                    last_user = history[i]["content"]
                    history = history[:i]  # removed; the send block re-appends it
                    break
            if last_user is None:
                print(f"{dim}(nothing to retry){reset}")
                continue
            user = last_user
        elif user.startswith("/"):
            print(f"{dim}(unknown command {user.split()[0]!r} — /help for the list){reset}")
            continue

        history.append({"role": "user", "content": user})
        reply = ""
        err = None
        answered = False        # have we printed any answer content yet?
        think_start = None      # monotonic clock when the first reasoning token arrived
        think_tokens = 0
        # Deep-think on → the selected model on the default path (Qwen3 reasons
        # by default). Off → the fast lane: fast_chat with thinking disabled. A
        # `think` field tells the gateway to use Ollama's native API, the only
        # surface that can actually suppress a reasoning model's thinking.
        if think_on:
            req = {"model": model, "messages": history, "stream": True}
        else:
            req = {"model": "fast_chat", "messages": history, "stream": True, "think": False}
        req["web_search"] = cfg.search.chat  # the engine adds its web tools only when asked
        spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        # Instant feedback: in deep-think mode the model usually reasons for a
        # beat before the first token, so seed the spinner now instead of
        # leaving a frozen prompt. (--show-thinking prints the reasoning itself;
        # the fast lane has no thinking to announce.)
        if think_on and not show_think:
            think_start = time.monotonic()
            sys.stdout.write(f"\r{dim}💭 thinking {spin[0]} 0 tok · 0s{reset}\033[K")
            sys.stdout.flush()
        try:
            with httpx.stream(
                "POST", f"{base}/v1/chat/completions",
                json=req,
                timeout=None,
            ) as r:
                if r.status_code != 200:
                    print(f"\n{dim}[error {r.status_code}] {r.read().decode()[:300]}{reset}",
                          file=sys.stderr)
                    history.pop()
                    continue
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except ValueError:
                        continue
                    # The gateway streams backend failures as an error frame
                    # over an already-200 response — surface it, don't swallow.
                    if obj.get("error"):
                        e = obj["error"]
                        err = e.get("message", str(e)) if isinstance(e, dict) else str(e)
                        break
                    # `choices` can be an empty list on the final/usage chunk —
                    # guard against IndexError (the [{}] default only applies
                    # when the key is absent, not when the list is empty).
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or ""
                    # Reasoning models (Qwen3, …) stream their chain of thought
                    # in a separate `reasoning` field with empty `content` — if
                    # we only watched `content`, the REPL would look frozen for
                    # the whole (often long) thinking phase. Surface it: a live
                    # in-place indicator by default, the full text with --show-thinking.
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if reasoning and not answered:
                        if think_start is None:
                            think_start = time.monotonic()
                            if show_think:
                                sys.stdout.write(f"{dim}💭 thinking…\n")
                        if show_think:
                            sys.stdout.write(reasoning)
                            sys.stdout.flush()
                        elif delta.get("hearth_tool"):
                            # A search step, not a token: name it rather than
                            # counting it, as these are the long silent stretches.
                            sys.stdout.write(f"\r{dim}{reasoning.strip()}{reset}\033[K")
                            sys.stdout.flush()
                        else:
                            think_tokens += 1
                            el = time.monotonic() - think_start
                            spinner = spin[think_tokens % len(spin)]
                            sys.stdout.write(
                                f"\r{dim}💭 thinking {spinner} {think_tokens} tok · {el:.0f}s{reset}\033[K")
                            sys.stdout.flush()
                    if content:
                        if not answered:
                            # Close the thinking indicator before the answer:
                            # full reasoning gets a blank line; the compact
                            # one-liner is wiped (\r + clear-to-end-of-line).
                            if think_start is not None:
                                sys.stdout.write(f"{reset}\n\n" if show_think else "\r\033[K")
                            sys.stdout.write(f"{bold}hearth>{reset} ")
                            answered = True
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        reply += content
        except (httpx.HTTPError, KeyboardInterrupt) as e:
            print(f"\n{dim}[interrupted: {e}]{reset}", file=sys.stderr)
            history.pop()
            continue
        # Finalize a thinking indicator that was never replaced by an answer
        # (empty reply, mid-stream error frame, or [DONE] with no content):
        # terminate the answer line if we printed one, else wipe the compact
        # one-liner / close the dim --show-thinking block so it neither lingers
        # nor bleeds colour into the next prompt.
        if answered:
            print()
        elif think_start is not None:
            sys.stdout.write(f"{reset}\n" if show_think else "\r\033[K")
            sys.stdout.flush()
        if err:
            print(f"{dim}[engine error: {err}]{reset}", file=sys.stderr)
            history.pop()  # drop the user turn that failed
        elif reply:
            history.append({"role": "assistant", "content": reply})
        else:
            history.pop()  # no content — don't pollute history with an empty turn
    return 0


# ---- control the background gateway service (systemd --user) -------------
#
# These subcommands drive a systemd --user unit that runs `hearth start`.
# Installing the unit is intentionally NOT exposed here: the desktop app
# (`hearth client install` → an icon that runs `hearth client open`) starts the gateway on
# demand, and `hearth start` / `hearth stop` cover manual control. So `service`
# operates on a unit that already exists (start/stop/restart/status/logs/autostart).

_UNIT_NAME = "hearth.service"


def _unit_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user" / _UNIT_NAME


def _sc_env() -> dict:
    # systemctl --user needs the user bus; supply XDG_RUNTIME_DIR if a
    # non-login shell didn't set it.
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return env


def _systemctl(*sc_args: str) -> int:
    sc = shutil.which("systemctl")
    if not sc:
        print("systemctl not found — there is no systemd user session here",
              file=sys.stderr)
        return 127
    return subprocess.run([sc, "--user", *sc_args], env=_sc_env()).returncode


def _service_autostart(on: bool, boot: bool) -> int:
    from . import install, service

    if not on:
        service.disable()
        print("hearth no longer starts at login")
    elif _unit_path().exists() and service.foreign():
        if service.systemctl("enable", _UNIT_NAME) != 0:  # the person's own unit: theirs to write, ours to enable
            return 1
        print("hearth starts at login")
    else:
        program = install.root() / "bin" / install.exe("hearthw" if sys.platform == "win32" else "hearth")
        if not install.managed() or not program.exists():
            print("only an installed hearth can be started at login (or a hearth.service of your own)", file=sys.stderr)
            return 1
        try:
            service.enable(str(program))
        except (service.Foreign, OSError) as e:
            print(e, file=sys.stderr)
            return 1
        print("hearth starts at login")
    lg = shutil.which("loginctl")
    if lg and (boot or not on):  # 'off' fully reverses 'on --boot'
        user = os.environ.get("USER") or str(os.getuid())
        subprocess.run([lg, "enable-linger" if on else "disable-linger", user], check=False)
    return 0


def _at_login() -> str:
    from . import service

    if _unit_path().exists() and service.foreign():
        return "started, by a unit of your own" if service.systemctl("is-enabled", "--quiet", _UNIT_NAME) == 0 else "not started"
    return "started" if service.enabled() else "not started"


def cmd_autostart(args: argparse.Namespace) -> int:
    if args.state is None:
        print(f"at login: {_at_login()}")
        return 0
    return _service_autostart(args.state == "on", args.boot)


def cmd_restart(args: argparse.Namespace) -> int:
    if _has_service() and _unit_runs():
        if (rc := _systemctl("stop", _UNIT_NAME)) != 0:
            return rc
        return cmd_start(build_parser().parse_args(["start"]))
    pid, _ = _engine_pid()  # answering or not
    if pid is None:
        if _gateway_identity(cfgmod.load()) is not None:
            print("a hearth answers, and its process was not found (another user's?): it was not restarted", file=sys.stderr)
            return 1
        return cmd_start(build_parser().parse_args(["start"]))
    # As it was started — and what could not be started again is not stopped.
    argv = procs.argv(pid)
    how = argv[argv.index("serve") + 1:] if "serve" in argv else []
    try:
        with contextlib.redirect_stderr(None):
            served = build_parser().parse_args(["serve", *how])
            start = build_parser().parse_args(["start", *how])
    except SystemExit:
        served = start = None
    host = cfgmod.load().bind_host
    if served is not None and served.bind:
        host = served.bind.rpartition(":")[0] or host
    if start is None or host not in _LOOPBACK:
        print("it was started with options of its own (hearth serve …): end that, and run it again", file=sys.stderr)
        return 2
    return cmd_stop(args) or cmd_start(start)


def cmd_logs(args: argparse.Namespace) -> int:
    """The service's journal, where that is what runs it; else the log file."""
    from . import background

    jc = shutil.which("journalctl")
    if jc and _has_service() and (_unit_runs() or _engine_pid()[0] is None):  # also when it failed: it is where it says why
        tail = ["-f"] if args.follow else ["-n", "200", "--no-pager"]
        return subprocess.run([jc, "--user", "-u", _UNIT_NAME, *tail], env=_sc_env()).returncode
    p = background.log_path()
    if not p.exists():
        print(f"no log yet ({p})")
        return 0
    with open(p, encoding="utf-8", errors="replace") as f:
        sys.stdout.writelines(deque(f, maxlen=200))
        try:
            while args.follow:
                sys.stdout.write(f.read())
                sys.stdout.flush()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return 0


# ---- the chat window: hearth-client -------------------------------------


def cmd_client(args: argparse.Namespace) -> int:
    """hearth's chat window, a program of its own (client.py). What is not about
    having it is its own command, and passed on."""
    from . import client, install
    from .provision import Problem

    rest = list(getattr(args, "rest", []))
    if rest[:1] not in (["install"], ["uninstall"]):
        # A chat window without an engine is of no use; the other commands do not need one.
        start = build_parser().parse_args(["start"])
        return client.open_window((lambda: cmd_start(start)) if rest in ([], ["open"]) else None, rest)
    p = argparse.ArgumentParser(prog=f"hearth client {rest[0]}")
    if rest[0] == "install":
        p.description = "Build the chat window, if it is not as it should be, and add it to the app menu."
        p.add_argument("--build", action="store_true", help="build it again")
    a = p.parse_args(rest[1:])
    try:
        if rest[0] == "install":
            install.ended_in_order()
            client.add(a.build, install.say)
        else:
            client.remove(install.say)
        return 0
    except (Problem, *install.ERRORS, httpx.HTTPError) as e:
        print(str(e) or type(e).__name__, file=sys.stderr)
        return 1


def cmd_update(args: argparse.Namespace) -> int:
    from . import install
    from .provision import Problem

    say = install.say
    install.ended_in_order()
    try:
        if args.rollback:
            prev = install.rollback_to()
            print(f"This goes back to hearth {prev['version']}, with the settings and chats from before the update. "
                  "Those of now are kept in a folder.")
            if not args.yes:
                try:
                    if input("Go back? [y/N] ").strip().lower() not in ("y", "yes"):
                        return 0
                except EOFError:
                    return 1
            install.rollback(say)
            return 0
        rel = install.available()
        if rel is None:
            print(f"hearth {install.current().get('version') or __version__} is the latest.")
            return 0
        print(f"hearth {rel.version} is available (you have {install.current().get('version') or __version__}).")
        if args.check:
            return 0
        if not args.yes:
            try:
                if input("Update now? [Y/n] ").strip().lower() in ("n", "no"):
                    return 0
            except EOFError:
                return 1
        install.update(rel, say)
        return 0
    except Problem as e:
        print(e, file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"The connection failed ({e or type(e).__name__}).", file=sys.stderr)
        return 1
    except install.ERRORS as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1


def cmd_uninstall(args: argparse.Namespace) -> int:
    from . import install
    from .provision import Problem

    keep = args.keep_models
    if not args.yes:
        try:
            if input("Remove hearth from this computer? [y/N] ").strip().lower() not in ("y", "yes"):
                return 0
            if not keep:
                keep = input("Keep the downloaded models and voices? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return 1
    try:
        install.uninstall(keep, args.keep_settings, lambda t: print(t, flush=True))
        return 0
    except (Problem, *install.ERRORS) as e:
        print(e, file=sys.stderr)
        return 1


_HELP = """\
AI on your own computer.

  hearth                      a coding agent in this folder (asks before it changes anything)
  hearth -p "…"               one task for it, without asking; its answer (-q: nothing else)
  hearth code FOLDER          the same for another folder, and its options
  hearth chat                 chat in the terminal
  hearth client               the chat window; `client install` puts it in the app menu

  hearth setup                the model that suits this computer, and natural voices
  hearth model …              list · suggest · install · use · remove
  hearth voice …              list · install · remove
  hearth provider …           list · add · remove: cloud or other servers

  hearth start · stop · restart · status · logs
  hearth autostart on|off     start the engine at login
  hearth serve                the engine in the foreground (what a service runs)

  hearth search "…"           the engine's web search, without a model
  hearth update · uninstall

`hearth <command> --help` has the rest."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hearth", usage="hearth [command] …", description=_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    p.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    # Bare `hearth` (no command) is the coding agent in this folder, like bare
    # `claude`. A command's own set_defaults(func=...) overrides these.
    p.set_defaults(func=cmd_code, model=None, path=".", yolo=False, max_rounds=60)
    p.add_argument("-p", "--prompt", help=argparse.SUPPRESS)
    p.add_argument("-q", "--quiet", action="store_true", help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", prog="hearth", metavar="command", help=argparse.SUPPRESS, required=False)

    def group(name: str, what: str):
        g = sub.add_parser(name, description=what)
        return g.add_subparsers(dest="action", required=True, metavar="command")

    # ── the engine
    for name, func, what in (("start", cmd_start, "Start the engine; it keeps running."),
                             ("serve", cmd_serve, "The engine in the foreground: what a service runs.")):
        s = sub.add_parser(name, description=what)
        s.add_argument("--bind", help="host:port override (default from config)")
        s.add_argument("--no-manage", action="store_true", help="don't start/stop Ollama")
        s.add_argument("--log-level", default="info")
        if name == "serve":
            s.add_argument("--unsafe-bind", action="store_true", help="allow non-loopback Host (requires --api-key)")
            s.add_argument("--api-key", help="bearer token required when --unsafe-bind")
        else:
            s.add_argument("--background", action="store_true", help=argparse.SUPPRESS)  # how apps say it
        s.set_defaults(func=func)
    sub.add_parser("stop", description="Stop the engine, its Ollama with it.").set_defaults(func=cmd_stop)
    sub.add_parser("restart", description="Stop it, and start it again.").set_defaults(func=cmd_restart)
    sub.add_parser("status", description="Whether the engine runs, and which model answers for what.").set_defaults(func=cmd_status)
    s = sub.add_parser("logs", description="The engine's log.")
    s.add_argument("-f", "--follow", action="store_true", help="and what is added to it")
    s.set_defaults(func=cmd_logs)
    s = sub.add_parser("autostart", description="Start the engine at login. Without on or off: whether it is.")
    s.add_argument("state", nargs="?", choices=["on", "off"])
    s.add_argument("--boot", action="store_true", help="with on: at boot already, before login (Linux)")
    s.set_defaults(func=cmd_autostart)

    # ── models
    m = group("model", "Models: what is here, what suits this computer, what is used.")
    s = m.add_parser("list", help="what is on this computer, and what it is used for")
    s.add_argument("--all", action="store_true", help="and what the providers have")
    s.set_defaults(func=cmd_model_list)
    s = m.add_parser("suggest", help="which models suit this computer, best first")
    s.add_argument("--json", action="store_true", help="with what was found out about the computer")
    s.set_defaults(func=cmd_model_suggest)
    s = m.add_parser("install", aliases=["add"], help="download a model, check that it works here, and use it")
    s.add_argument("model")
    s.add_argument("--anyway", action="store_true", help="use it although it changed, or failed a check")
    s.add_argument("--download-only", action="store_true", help="neither check nor use it")
    s.set_defaults(func=cmd_model_install)
    s = m.add_parser("use", help="which model answers (the chat and the agent, or --for a role)")
    s.add_argument("model")
    s.add_argument("--for", dest="roles", action="append", choices=cfgmod.ROLE_NAMES, metavar="ROLE",
                   help=f"one of {', '.join(cfgmod.ROLE_NAMES)}; may be given more than once")
    s.add_argument("--provider", help="where the model is, if not on this computer")
    s.add_argument("--context", type=int, default=None, help="how many tokens it gets (default: measured)")
    s.add_argument("--anyway", action="store_true", help="although it changed, or failed a check")
    s.set_defaults(func=cmd_model_use)
    s = m.add_parser("remove", help="delete a model's files (not while it is used)")
    s.add_argument("model")
    s.set_defaults(func=cmd_model_remove)
    s = sub.add_parser("setup", description="Install the model that suits this computer, and natural voices.")
    s.add_argument("--yes", "-y", action="store_true", help="don't ask")
    s.add_argument("--no-voice", action="store_true", help="without the voices")
    s.set_defaults(func=cmd_setup)

    # ── voices
    v = group("voice", "Voices for reading aloud.")
    v.add_parser("list", help="what is here").set_defaults(func=cmd_voice_list)
    s = v.add_parser("install", help="get one")
    s.add_argument("name", help=f"'{_KOKORO}' (the natural voices), or a Piper voice such as {_DEFAULT_VOICE}")
    s.set_defaults(func=cmd_voice_install)
    s = v.add_parser("remove", help="delete one")
    s.add_argument("name")
    s.set_defaults(func=cmd_voice_remove)

    # ── providers
    b = group("provider", "Where models are, besides this computer: a cloud, or another server that speaks OpenAI's API.")
    b.add_parser("list", help="which there are").set_defaults(func=cmd_provider)
    s = b.add_parser("add", help="add one")
    s.add_argument("name")
    s.add_argument("--type", default="openai", choices=["ollama", "openai"])
    s.add_argument("--base-url", required=True, dest="base_url",
                   help="versioned root, e.g. https://open.bigmodel.cn/api/paas/v4")
    s.add_argument("--api-key", dest="api_key", help="literal API key (prefer --api-key-env)")
    s.add_argument("--api-key-env", dest="api_key_env",
                   help="name of an env var holding the API key (keeps it out of config)")
    s.set_defaults(func=cmd_provider)
    s = b.add_parser("remove", help="take one out")
    s.add_argument("name")
    s.set_defaults(func=cmd_provider)

    # ── talking to it
    s = sub.add_parser("chat", description="Chat in the terminal.")
    s.add_argument("--model", help="role alias or model id (default: primary_chat)")
    s.add_argument("--system", help="optional system prompt")
    s.add_argument("--no-think", action="store_true",
                   help="start with deep-think off (fast lane); toggle in-chat with /think on|off")
    s.add_argument("--show-thinking", action="store_true",
                   help="show the model's full reasoning text (default: compact indicator)")
    s.set_defaults(func=cmd_chat)
    s = sub.add_parser("code", description="The coding agent, as plain `hearth`, for a folder: it reads, and asks "
                                           "before it changes a file, runs a command or goes to the web.")
    s.add_argument("path", nargs="?", default=".", help="the folder (default: this one)")
    s.add_argument("-p", "--prompt", help="one task, without asking; its answer, and exit")
    s.add_argument("-q", "--quiet", action="store_true", help="with -p: only the final answer, no tool trace")
    s.add_argument("--model", help="role alias or model id (default: coding)")
    s.add_argument("--max-rounds", type=int, default=60,
                   help="tool rounds before the agent stops and asks (default 60)")
    s.add_argument("--yolo", action="store_true", help="do not ask before changes, commands and the web")
    s.set_defaults(func=cmd_code)
    s = sub.add_parser("search", description="Query the web through the engine's search chain.")
    s.add_argument("query", help="search query, or a URL with --fetch")
    s.add_argument("-n", "--count", type=int, default=0, help="number of results")
    s.add_argument("--provider", choices=["auto", "brave", "searxng", "brave_html", "wikipedia"],
                   help="force one provider instead of the configured chain")
    s.add_argument("--fetch", action="store_true",
                   help="treat the argument as a URL and print the page text")
    s.set_defaults(func=cmd_search)
    # (The window's arguments are its own: main() passes them on unread.)
    sub.add_parser("client", add_help=False).set_defaults(func=cmd_client, rest=[])

    # ── hearth itself
    s = sub.add_parser("update", description="Update hearth and its chat window, or go back.")
    s.add_argument("--check", action="store_true", help="only say whether there is one")
    s.add_argument("--yes", "-y", action="store_true", help="don't ask")
    s.add_argument("--rollback", action="store_true", help="go back to the version before")
    s.set_defaults(func=cmd_update)
    s = sub.add_parser("uninstall", description="Remove hearth from this computer.")
    s.add_argument("--yes", "-y", action="store_true", help="don't ask")
    s.add_argument("--keep-models", action="store_true", help="keep downloaded models and voices")
    s.add_argument("--keep-settings", action="store_true", help="keep ~/.hearth")
    s.set_defaults(func=cmd_uninstall)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["client"]:
        return cmd_client(argparse.Namespace(rest=argv[1:]))
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
