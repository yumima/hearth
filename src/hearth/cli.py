"""hearth CLI — start, models, pull, bind, swap, roles, hardware.

`hearth start` is the one-shot: it brings up the bundled Ollama (if not
already running), then serves the OpenAI-compatible gateway on loopback.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
from pathlib import Path

import httpx

from . import config as cfgmod
from . import hardware, ollama_supervisor


def _admin_base(cfg: cfgmod.Config) -> str:
    return f"http://{cfg.bind_host}:{cfg.bind_port}"


def _pid_path() -> Path:
    """PID file for the running gateway, alongside config.yaml (~/.hearth)."""
    return cfgmod.config_path().parent / "hearth.pid"


def cmd_start(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app

    cfg = cfgmod.load()
    if args.bind:
        host, _, port = args.bind.rpartition(":")
        cfg.bind_host, cfg.bind_port = host or cfg.bind_host, int(port)

    allow_any_host = bool(args.unsafe_bind)
    api_key = args.api_key
    if allow_any_host and not api_key:
        print("--unsafe-bind requires --api-key (off-loopback exposure)", file=sys.stderr)
        return 2

    # Bring up Ollama unless told not to.
    ollama = cfg.backends.get("ollama")
    if ollama and not args.no_manage:
        proc = ollama_supervisor.start(ollama.base_url)
        if proc is not None:
            print(f"[ollama] started ({ollama.base_url})")
        elif ollama_supervisor.is_up(ollama.base_url):
            print(f"[ollama] already running ({ollama.base_url})")
        else:
            print(
                f"[ollama] WARNING not running and no binary found at {ollama.base_url}; "
                "chat/embeddings will be unavailable until it's up",
                file=sys.stderr,
            )

    # Write a PID file so `hearth stop` / `hearth status` can find this gateway.
    pid_file = _pid_path()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    atexit.register(lambda: pid_file.unlink(missing_ok=True))

    app = create_app(cfg, allow_any_host=allow_any_host, api_key=api_key)
    print(f"[hearth] serving OpenAI-compatible API on http://{cfg.bind_host}:{cfg.bind_port}")
    print(f"[hearth] probe: GET /v1/models   health: GET /admin/health   chat: hearth chat")
    # server_header=False suppresses uvicorn's own "Server: uvicorn" so our
    # middleware's "Server: hearth/<v>" is the single, clean identity.
    uvicorn.run(app, host=cfg.bind_host, port=cfg.bind_port,
                log_level=args.log_level, server_header=False)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    base = _admin_base(cfg)
    # Prefer the running gateway; fall back to Ollama directly.
    try:
        r = httpx.get(f"{base}/v1/models", timeout=5.0)
        data = r.json().get("data", [])
        for m in data:
            print(f"{m['id']:40s} {m.get('owned_by','')}")
        return 0
    except httpx.HTTPError:
        pass
    ollama = cfg.backends.get("ollama")
    if ollama and ollama_supervisor.is_up(ollama.base_url):
        r = httpx.get(f"{ollama.base_url}/api/tags", timeout=5.0)
        for m in r.json().get("models", []):
            print(m.get("name", ""))
        return 0
    print("no running gateway or ollama daemon", file=sys.stderr)
    return 1


def cmd_pull(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    ollama = cfg.backends.get("ollama")
    if not ollama:
        print("no ollama backend configured", file=sys.stderr)
        return 1
    if not ollama_supervisor.is_up(ollama.base_url):
        proc = ollama_supervisor.start(ollama.base_url)
        if proc is None and not ollama_supervisor.is_up(ollama.base_url):
            print("could not start ollama", file=sys.stderr)
            return 1
    print(f"pulling {args.model} ...")
    last = ""
    with httpx.stream(
        "POST", f"{ollama.base_url}/api/pull",
        json={"model": args.model, "stream": True}, timeout=None,
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
                return 1
    print("\ndone")
    return 0


def cmd_bind(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    if args.role not in cfgmod.ROLE_NAMES:
        print(f"unknown role {args.role!r}; valid: {', '.join(cfgmod.ROLE_NAMES)}", file=sys.stderr)
        return 2
    cfg.roles[args.role] = cfgmod.RoleBinding(model=args.model, backend=args.backend)
    cfgmod.save(cfg)
    print(f"bound {args.role} -> {args.model} ({args.backend})")
    # If a gateway is running, hot-apply via admin so no restart is needed.
    base = _admin_base(cfg)
    try:
        httpx.put(
            f"{base}/admin/roles/{args.role}",
            json={"model": args.model, "backend": args.backend}, timeout=3.0,
        )
        print("(applied to running gateway)")
    except httpx.HTTPError:
        pass
    return 0


def cmd_roles(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    for name, r in cfg.roles.items():
        print(f"{name:14s} -> {r.model} ({r.backend})")
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    print(json.dumps(hardware.as_dict(), indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    base = _admin_base(cfg)
    try:
        ver = httpx.get(f"{base}/admin/version", timeout=3.0).json()
        health = httpx.get(f"{base}/admin/health", timeout=3.0).json()
    except httpx.HTTPError:
        print(f"hearth: NOT running at {base}  (start with `hearth start`)")
        return 1
    print(f"hearth {ver.get('version')}  (contract {ver.get('contract')})  @ {base}")
    print(f"status: {health.get('status')}   backends: {health.get('backends')}")
    try:
        roles = httpx.get(f"{base}/admin/roles", timeout=3.0).json().get("roles", {})
        for k, v in roles.items():
            print(f"  {k:14s} -> {v['model']}")
    except httpx.HTTPError:
        pass
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    pid_file = _pid_path()
    if not pid_file.exists():
        print("no hearth pidfile — gateway not started by this CLI (or already stopped)",
              file=sys.stderr)
        return 1
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        print("invalid pidfile (removed)", file=sys.stderr)
        return 1
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"stopped hearth gateway (pid {pid})")
    except ProcessLookupError:
        print("hearth gateway not running (stale pidfile)")
    pid_file.unlink(missing_ok=True)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """A tiny streaming REPL against the local gateway — like ChatGPT in the
    terminal, but fully local."""
    cfg = cfgmod.load()
    base = _admin_base(cfg)
    try:
        httpx.get(f"{base}/admin/health", timeout=3.0)
    except httpx.HTTPError:
        print(f"hearth gateway not reachable at {base} — run `hearth start` first",
              file=sys.stderr)
        return 1

    model = args.model or "primary_chat"
    bold, dim, reset = "\033[1m", "\033[2m", "\033[0m"
    print(f"{dim}hearth chat — model={model}  (/exit to quit, /reset to clear, /model NAME to switch){reset}")
    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    while True:
        try:
            user = input(f"\n{bold}you>{reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print(f"{dim}(history cleared){reset}")
            continue
        if user.startswith("/model "):
            model = user.split(maxsplit=1)[1].strip()
            print(f"{dim}(model -> {model}){reset}")
            continue

        history.append({"role": "user", "content": user})
        sys.stdout.write(f"{bold}hearth>{reset} ")
        sys.stdout.flush()
        reply = ""
        try:
            with httpx.stream(
                "POST", f"{base}/v1/chat/completions",
                json={"model": model, "messages": history, "stream": True},
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
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        reply += delta
        except (httpx.HTTPError, KeyboardInterrupt) as e:
            print(f"\n{dim}[interrupted: {e}]{reset}", file=sys.stderr)
            history.pop()
            continue
        print()
        history.append({"role": "assistant", "content": reply})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hearth", description="Local OpenAI-compatible AI engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start", help="serve the gateway (brings up Ollama)")
    s.add_argument("--bind", help="host:port override (default from config)")
    s.add_argument("--no-manage", action="store_true", help="don't start/stop Ollama")
    s.add_argument("--unsafe-bind", action="store_true", help="allow non-loopback Host (requires --api-key)")
    s.add_argument("--api-key", help="bearer token required when --unsafe-bind")
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_start)

    s = sub.add_parser("models", help="list servable models")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("pull", help="pull a model via Ollama")
    s.add_argument("model")
    s.set_defaults(func=cmd_pull)

    s = sub.add_parser("bind", help="bind a role to a model")
    s.add_argument("role")
    s.add_argument("model")
    s.add_argument("--backend", default="ollama")
    s.set_defaults(func=cmd_bind)
    # `swap` is an alias for hot-rebind.
    s2 = sub.add_parser("swap", help="alias for bind (hot rebind)")
    s2.add_argument("role")
    s2.add_argument("model")
    s2.add_argument("--backend", default="ollama")
    s2.set_defaults(func=cmd_bind)

    s = sub.add_parser("roles", help="show role bindings")
    s.set_defaults(func=cmd_roles)

    s = sub.add_parser("hardware", help="print hardware probe")
    s.set_defaults(func=cmd_hardware)

    s = sub.add_parser("status", help="show whether the gateway is up + roles")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("stop", help="stop a running gateway (started via `hearth start`)")
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("chat", help="interactive streaming chat with the local engine")
    s.add_argument("--model", help="role alias or model id (default: primary_chat)")
    s.add_argument("--system", help="optional system prompt")
    s.set_defaults(func=cmd_chat)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
