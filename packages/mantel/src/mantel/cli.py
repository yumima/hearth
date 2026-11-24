"""mantel CLI.

  mantel            open the desktop chat window (starts mantel's local server)
  mantel serve      run the local server in the foreground (headless / dev)
  mantel stop       stop the local server
  mantel restart    restart the local server
  mantel logs [-f]  show (or follow) the local server log
  mantel install    add a clickable desktop launcher (app menu / Start menu)
  mantel uninstall  remove the launcher

mantel is a *client*: bare `mantel` opens the window. The window runs mantel's
own loopback server (a daemon thread) which serves the UI and proxies to the
configured backend (default: local hearth on :11435).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from . import config as cfgmod


def _server_url(cfg: cfgmod.Config) -> str:
    return f"http://{cfg.host}:{cfg.port}/"


def _server_up(cfg: cfgmod.Config, timeout: float = 0.5) -> bool:
    """True only if *mantel* is serving on the port — not some other app that
    happens to answer /api/health (so we never reuse a foreign server)."""
    try:
        r = httpx.get(f"http://{cfg.host}:{cfg.port}/api/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("app") == "mantel"
    except (httpx.HTTPError, ValueError):
        return False


def _read_pid() -> int | None:
    """The PID of a live mantel server from the pidfile, or None if the file is
    absent/garbage or the process is gone (stale pidfile)."""
    try:
        pid = int(cfgmod.pid_path().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)  # signal 0 = existence check, doesn't actually signal
    except ProcessLookupError:
        return None  # stale
    except PermissionError:
        return pid  # exists but not ours (shouldn't happen on loopback/same user)
    return pid


def _pid_is_mantel(pid: int) -> bool:
    """Best-effort check that `pid` really is a mantel process, so a stale pidfile
    whose PID the OS has since recycled can't make `stop` signal an unrelated
    process. A hung mantel still matches (its cmdline is unchanged), so this
    doesn't block killing one. On platforms without /proc we can't tell — assume
    yes and fall back to the pidfile's word."""
    try:
        return b"mantel" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True


def _open_server_log():
    """Open the server log for append, rotating once when it grows past ~2 MB so a
    long-lived daemon can't fill the disk. Returns a file object (or None on error,
    in which case the caller should fall back to DEVNULL)."""
    lp = cfgmod.log_path()
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        if lp.exists() and lp.stat().st_size > 2_000_000:
            lp.replace(lp.with_name(lp.name + ".1"))
        return open(lp, "a", buffering=1, encoding="utf-8")
    except OSError:
        return None


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app

    cfg = cfgmod.load()
    print(f"[mantel] serving http://{cfg.host}:{cfg.port}  (backend: {cfg.active().base_url})")
    print(f"[mantel] open the UI at {_server_url(cfg)}")

    # Record our PID so `mantel stop`/`restart` can find this server (works whether
    # we were launched in the foreground or detached by _ensure_server). Remove it
    # on exit, but only if it's still ours — a racing restart may have replaced it.
    pid_file = cfgmod.pid_path()
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pid_file = None  # non-fatal: serve anyway, just not stoppable via pidfile

    try:
        uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port,
                    log_level=getattr(args, "log_level", "info"))
    finally:
        if pid_file is not None:
            try:
                if pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    pid_file.unlink()
            except OSError:
                pass
    return 0


def _ensure_server(cfg: cfgmod.Config) -> bool:
    """Ensure mantel's server is up, starting it as a DETACHED process if needed.

    The server must outlive the launcher. Chrome's single-instance reuse makes the
    spawned browser process exit immediately when a window for the profile already
    exists (e.g. a leftover Chrome from a prior session), and the previous
    in-process daemon-thread server died with it — the freshly-opened window then
    loaded a dead port ("first launch refused, second works"). A detached
    `mantel serve` survives independently and is reused on the next open, so the
    window always finds a live server. Returns True once it's serving."""
    if _server_up(cfg):
        return True
    # Send the detached server's output to a log file (was DEVNULL — which left no
    # way to see why a start failed). `mantel logs` tails it.
    log = _open_server_log()
    out = log or subprocess.DEVNULL
    try:
        subprocess.Popen([sys.executable, "-m", "mantel", "serve"],
                         start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    finally:
        if log is not None:
            log.close()  # child inherited the fd; parent doesn't need its copy
    # Cold start can take a while (fresh .pyc + the MCP SDK import); poll until up.
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if _server_up(cfg):
            return True
        time.sleep(0.2)
    return False


def cmd_open(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    if not _ensure_server(cfg):
        print(f"mantel's server didn't come up on {cfg.host}:{cfg.port}. "
              f"Run `mantel serve` to see the error, or change `port` via "
              f"`mantel config edit`.", file=sys.stderr)
        return 1
    from . import desktop
    return desktop.open_window(_server_url(cfg))


def cmd_stop(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    pid = _read_pid()
    if pid is None:
        cfgmod.pid_path().unlink(missing_ok=True)  # clear a stale pidfile
        if _server_up(cfg):
            print("a mantel server is running but no pid file was found "
                  "(started by an older build?); stop it manually.", file=sys.stderr)
            return 1
        print("mantel is not running.")
        return 0
    if not _pid_is_mantel(pid):
        cfgmod.pid_path().unlink(missing_ok=True)  # PID was recycled — not our server
        print("mantel is not running (cleared a stale pid file).")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        cfgmod.pid_path().unlink(missing_ok=True)
        print("mantel is not running.")
        return 0
    # Wait for graceful shutdown; escalate to SIGKILL if it won't go.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        try:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))  # SIGKILL is POSIX-only
        except ProcessLookupError:
            pass
    cfgmod.pid_path().unlink(missing_ok=True)
    print(f"stopped mantel (pid {pid}).")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    cmd_stop(args)
    cfg = cfgmod.load()
    if _ensure_server(cfg):
        print(f"mantel restarted on {cfg.host}:{cfg.port}  ·  logs: {cfgmod.log_path()}")
        return 0
    print(f"mantel didn't come back up on {cfg.host}:{cfg.port} — see {cfgmod.log_path()}",
          file=sys.stderr)
    return 1


def cmd_logs(args: argparse.Namespace) -> int:
    lp = cfgmod.log_path()
    if not lp.exists():
        print(f"no log yet at {lp}  (start the server with `mantel` or `mantel serve`).")
        return 0
    if getattr(args, "follow", False):
        tail = shutil.which("tail")
        if tail:
            return subprocess.call([tail, "-n", "200", "-f", str(lp)])
        print("(`tail` not found — showing the last 200 lines instead)\n", file=sys.stderr)
    # Print the tail without slurping a large file into memory.
    from collections import deque
    with open(lp, encoding="utf-8", errors="replace") as f:
        for line in deque(f, maxlen=200):
            sys.stdout.write(line)
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    from . import desktop
    return desktop.install(_mantel_exec_argv())


def cmd_uninstall(args: argparse.Namespace) -> int:
    from . import desktop
    return desktop.uninstall()


def _mantel_exec_argv() -> list[str]:
    """argv a launcher uses to invoke mantel — bare `mantel` opens the window.
    Prefers a `mantel` on PATH, else the console script next to this interpreter
    (the venv's bin/mantel), routed through a stable ~/.local/bin/mantel symlink.
    Falls back to ``<this python> -m mantel``.

    NB: do NOT realpath the interpreter — a venv python is a symlink to the base
    python, which doesn't have mantel (or its deps) installed."""
    found = shutil.which("mantel")
    if not found:
        cand = Path(sys.executable).with_name("mantel")
        if cand.exists():
            found = str(cand)
    if found:
        return [_stable_link(os.path.realpath(found))]
    return [sys.executable, "-m", "mantel"]


def _stable_link(real: str) -> str:
    link = Path.home() / ".local" / "bin" / "mantel"
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() and not link.is_symlink():
            return str(link)  # a real file already sits there — trust it
        if not link.exists() or os.path.realpath(link) != real:
            if link.is_symlink():
                link.unlink()
            link.symlink_to(real)
        return str(link)
    except OSError:
        return real


# ── environment management (hearth-style) ─────────────────────────────────────


def _probe_models(base_url: str) -> tuple[bool, list[dict]]:
    """(reachable, models) from an OpenAI-compatible /models endpoint."""
    try:
        r = httpx.get(f"{base_url}/models", timeout=4.0)
        if r.status_code == 200:
            return True, (r.json().get("data") or [])
    except (httpx.HTTPError, ValueError):
        pass
    return False, []


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _enumerate_tools(cfg: cfgmod.Config) -> tuple[list[dict], dict]:
    """Start the configured MCP servers transiently to list their tools."""
    if not cfg.mcp_servers:
        return [], {}
    import asyncio

    from .mcp_host import MCPHost

    async def go():
        h = MCPHost(cfg.mcp_servers)
        await h.start()
        tools = [{"name": t["function"]["name"], "description": t["function"]["description"]}
                 for t in h.openai_tools()]
        errs = dict(h.errors)
        await h.stop()
        return tools, errs

    try:
        return asyncio.run(go())
    except Exception as e:  # noqa: BLE001
        return [], {"_": str(e)}


def _live_or_transient_tools(cfg: cfgmod.Config) -> tuple[list[dict], dict]:
    if _server_up(cfg):
        try:
            j = httpx.get(f"http://{cfg.host}:{cfg.port}/api/tools", timeout=5.0).json()
            return j.get("tools", []), j.get("errors", {})
        except (httpx.HTTPError, ValueError):
            pass
    return _enumerate_tools(cfg)


def _example_mcp_servers() -> dict:
    """Safe, **disabled** servers to seed the config: two Node/npx examples plus
    mantel's built-in RAG server (local embeddings-backed search over your files)."""
    from .mcp_host import ServerConfig
    home = str(Path.home())
    return {
        "filesystem": ServerConfig(
            command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", home],
            enabled=False, auto_approve=["read_*", "list_*", "directory_*", "search_*"]),
        "memory": ServerConfig(
            command="npx", args=["-y", "@modelcontextprotocol/server-memory"],
            enabled=False, auto_approve=["*"]),
        # Built-in: RAG over the user's files via the backend's embeddings.
        # sys.executable is the venv python running mantel → `import mantel` works.
        "rag": ServerConfig(
            command=sys.executable, args=["-m", "mantel.mcp_servers.rag"],
            enabled=False, auto_approve=["*"]),
        # Built-in: fetch a web page/API by URL (readable text).
        "web": ServerConfig(
            command=sys.executable, args=["-m", "mantel.mcp_servers.web"],
            enabled=False, auto_approve=["*"]),
        # Built-in: text-to-image (ComfyUI/SDXL via hearth /v1/images/generations).
        "image": ServerConfig(
            command=sys.executable, args=["-m", "mantel.mcp_servers.image"],
            enabled=False, auto_approve=["*"]),
    }


def cmd_setup(args: argparse.Namespace) -> int:
    """First-run wizard: find a backend, pick a model, seed MCP examples."""
    cfg = cfgmod.load()
    print("mantel setup\n")
    url = cfgmod.DEFAULT_HEARTH_URL
    print(f"Looking for a local hearth engine at {url} …")
    ok, models = _probe_models(url)
    if ok:
        cfg.provider = "hearth"
        cfg.providers["hearth"] = cfgmod.Provider(base_url=url)
        ids = [m.get("id") for m in models]
        cfg.model = "primary_chat" if "primary_chat" in ids else (ids[0] if ids else "primary_chat")
        print(f"  ✓ found hearth ({len(models)} models) — using model '{cfg.model}'")
    else:
        print("  · none found. Run a local hearth (github.com/yumima/hearth), or edit the")
        print("    config to point at a cloud backend (`mantel config edit`).")

    if not cfg.mcp_servers and shutil.which("npx"):
        cfg.mcp_servers = _example_mcp_servers()
        print("\nMCP: `npx` found — seeded two safe, disabled example servers")
        print("     (filesystem, memory). Enable with `mantel mcp enable <name>`.")
    elif not cfg.mcp_servers:
        print("\nMCP: no `npx`/`uvx` runtime found — add servers later with `mantel config edit`.")

    p = cfgmod.save(cfg)
    print(f"\n✓ wrote {p}")
    try:
        import webview  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001
        print("\nTip: for a native window (instead of a Chromium --app window), install")
        print("     pywebview:  pip install 'mantel[gui]'  (the chrome fallback works without it)")
    print("\nNext:  mantel        # open the chat window")
    print("       mantel status # check backend + tools")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    prov = cfg.active()
    ok, models = _probe_models(prov.base_url)
    print(f"mantel  ·  config: {cfgmod.config_path()}"
          f"{'' if cfgmod.config_path().exists() else '  (defaults)'}")
    pid = _read_pid()
    print(f"  server:  http://{cfg.host}:{cfg.port}  "
          f"({('running, pid ' + str(pid)) if _server_up(cfg) else 'not running'})")
    if _server_up(cfg):
        print(f"  logs:    {cfgmod.log_path()}")
    print(f"  backend: {cfg.provider} → {prov.base_url}  "
          f"({str(len(models)) + ' models' if ok else 'unreachable'})")
    print(f"  model:   {cfg.model}")
    if not cfg.mcp_servers:
        print("  mcp:     no servers (run `mantel setup` or `mantel config edit`)")
    else:
        enabled = [n for n, s in cfg.mcp_servers.items() if s.enabled]
        print(f"  mcp:     {len(cfg.mcp_servers)} server(s), {len(enabled)} enabled"
              + (f": {', '.join(enabled)}" if enabled else ""))
        if enabled:
            tools, errs = _live_or_transient_tools(cfg)
            print(f"  tools:   {len(tools)} available"
                  + (f"  ·  errors: {', '.join(errs)}" if errs else ""))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    print("mantel doctor — environment check\n")
    ok, models = _probe_models(cfg.active().base_url)
    print(f"  [{_mark(ok)}] backend  {cfg.provider} ({cfg.active().base_url})"
          + (f" — {len(models)} models" if ok else " — unreachable"))
    for rt, why in (("node", "needed for npx servers"), ("npx", "Node MCP servers"),
                    ("uvx", "Python MCP servers (uv)")):
        path = shutil.which(rt)
        print(f"  [{_mark(bool(path))}] {rt:5} {('— ' + why):36}{path or 'missing'}")
    try:
        import webview  # type: ignore  # noqa: F401
        have_pw = True
    except Exception:  # noqa: BLE001
        have_pw = False
    have_chrome = any(shutil.which(b) for b in
                      ("google-chrome", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"))
    print(f"  [{_mark(have_pw)}] window   pywebview (native)"
          + ("" if have_pw else "  — pip install 'mantel[gui]'"))
    print(f"  [{_mark(have_chrome)}] window   chromium/edge --app fallback")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    ok, models = _probe_models(cfg.active().base_url)
    if not ok:
        print(f"backend unreachable at {cfg.active().base_url}", file=sys.stderr)
        return 1
    for m in models:
        print(f"  {m.get('id', ''):40s} {m.get('owned_by', '')}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    p = cfgmod.config_path()
    action = getattr(args, "action", None)
    if action == "path":
        print(p)
        return 0
    if action == "edit":
        if not p.exists():
            cfgmod.save(cfgmod.load())  # materialize defaults so there's something to edit
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
        return subprocess.call([editor, str(p)])
    if p.exists():
        print(f"# {p}\n{p.read_text(encoding='utf-8').rstrip()}")
    else:
        print(f"# {p}  (defaults — not written yet; `mantel setup` or `mantel config edit` to create)")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    action = args.action
    if action == "list":
        if not cfg.mcp_servers:
            print("no MCP servers configured  (run `mantel setup`).")
            return 0
        for n, s in cfg.mcp_servers.items():
            print(f"  {'●' if s.enabled else '○'} {n:14s} {s.command} {' '.join(s.args)}")
            if s.auto_approve:
                print(f"      auto-approve: {', '.join(s.auto_approve)}")
        return 0
    if action == "tools":
        tools, errs = _live_or_transient_tools(cfg)
        for t in tools:
            print(f"  {t['name']:34s} {t.get('description', '')[:60]}")
        for n, e in errs.items():
            print(f"  ! {n}: {e}", file=sys.stderr)
        if not tools and not errs:
            print("no tools — enable a server with `mantel mcp enable <name>`.")
        return 0
    if action in ("enable", "disable"):
        if args.name not in cfg.mcp_servers:
            print(f"no such server '{args.name}'  (see `mantel mcp list`)", file=sys.stderr)
            return 1
        cfg.mcp_servers[args.name].enabled = (action == "enable")
        cfgmod.save(cfg)
        print(f"✓ {args.name} {action}d — restart mantel to apply.")
        return 0
    print(f"unknown mcp action {action!r}", file=sys.stderr)
    return 2


def cmd_provider(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    action = args.action
    if action == "list":
        for n, pr in cfg.providers.items():
            print(f"  {'●' if n == cfg.provider else '○'} {n:12s} {pr.base_url}")
        return 0
    if action == "use":
        if args.name not in cfg.providers:
            print(f"no such provider '{args.name}'  (see `mantel provider list`)", file=sys.stderr)
            return 1
        cfg.provider = args.name
        cfgmod.save(cfg)
        print(f"✓ active provider → {args.name} — restart mantel to apply.")
        return 0
    print(f"unknown provider action {action!r}", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mantel",
        description="Universal, MCP-native chat client — defaults to a local hearth engine.")
    p.set_defaults(func=cmd_open)  # bare `mantel` opens the window
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("setup", help="first-run wizard — find a backend, pick a model, seed MCP examples") \
       .set_defaults(func=cmd_setup)

    s = sub.add_parser("serve", help="run the local server in the foreground (headless / dev)")
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)

    sub.add_parser("open", help="open the desktop chat window").set_defaults(func=cmd_open)
    sub.add_parser("stop", help="stop the local server").set_defaults(func=cmd_stop)
    sub.add_parser("restart", help="restart the local server (stop, then start detached)") \
       .set_defaults(func=cmd_restart)
    lg = sub.add_parser("logs", help="show the local server log")
    lg.add_argument("-f", "--follow", action="store_true", help="follow the log (tail -f)")
    lg.set_defaults(func=cmd_logs)
    sub.add_parser("status", help="backend reachability, model, MCP servers + tools").set_defaults(func=cmd_status)
    sub.add_parser("doctor", help="diagnose the environment (backend, MCP runtimes, window)").set_defaults(func=cmd_doctor)
    sub.add_parser("models", help="list models from the active backend").set_defaults(func=cmd_models)

    cf = sub.add_parser("config", help="show / edit the config file")
    cf.add_argument("action", nargs="?", choices=["show", "path", "edit"], default="show")
    cf.set_defaults(func=cmd_config)

    mp = sub.add_parser("mcp", help="manage MCP servers")
    mp_sub = mp.add_subparsers(dest="action", required=True)
    mp_sub.add_parser("list", help="list configured servers")
    mp_sub.add_parser("tools", help="list the tools the enabled servers expose")
    for act in ("enable", "disable"):
        sp = mp_sub.add_parser(act, help=f"{act} a server")
        sp.add_argument("name")
    mp.set_defaults(func=cmd_mcp)

    pv = sub.add_parser("provider", help="manage backends")
    pv_sub = pv.add_subparsers(dest="action", required=True)
    pv_sub.add_parser("list", help="list providers")
    pv_use = pv_sub.add_parser("use", help="set the active provider")
    pv_use.add_argument("name")
    pv.set_defaults(func=cmd_provider)

    sub.add_parser("install", help="install the clickable desktop app (app menu / Start menu)") \
       .set_defaults(func=cmd_install)
    sub.add_parser("uninstall", help="remove the desktop app launcher") \
       .set_defaults(func=cmd_uninstall)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
