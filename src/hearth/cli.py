"""hearth CLI — start, models, pull, bind, swap, roles, hardware.

`hearth start` is the one-shot: it brings up the bundled Ollama (if not
already running), then serves the OpenAI-compatible gateway on loopback.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from . import config as cfgmod
from . import hardware, ollama_supervisor


def _admin_base(cfg: cfgmod.Config) -> str:
    return f"http://{cfg.bind_host}:{cfg.bind_port}"


def _pid_path() -> Path:
    """PID file for the running gateway, alongside config.yaml (~/.hearth)."""
    return cfgmod.config_path().parent / "hearth.pid"


def _pid_looks_like_hearth(pid: int) -> bool:
    """Best-effort check that `pid` is a hearth gateway before we signal it,
    so a stale pidfile (after SIGKILL/crash) + PID reuse can't kill an
    unrelated process. On non-Linux (no /proc) we can't introspect — trust the
    pidfile."""
    proc = Path(f"/proc/{pid}/cmdline")
    if not proc.exists():
        return True
    try:
        cl = proc.read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        return True
    return "hearth" in cl or "uvicorn" in cl


def _pid_on_port(port: int) -> int | None:
    """Best-effort: PID listening on `port`, so `stop` can find a gateway that
    wasn't started via this CLI (no pidfile) — e.g. `python -m hearth start`.
    Tries `ss` then `lsof`; returns None if neither is available or nothing is
    listening. Without elevated privileges this only sees same-user processes,
    which is all we need (single-user, loopback)."""
    if shutil.which("ss"):
        try:
            out = subprocess.run(["ss", "-ltnHp", f"sport = :{port}"],
                                 capture_output=True, text=True, timeout=4.0, check=False)
            m = re.search(r"pid=(\d+)", out.stdout)
            if m:
                return int(m.group(1))
        except (OSError, subprocess.TimeoutExpired):
            pass
    if shutil.which("lsof"):
        try:
            out = subprocess.run(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True, timeout=4.0, check=False)
            for tok in out.stdout.split():
                if tok.isdigit():
                    return int(tok)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


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

    # If hearth is installed as a service, a bare `hearth start` at a terminal is
    # a FOREGROUND instance — not the daemon. Nudge the user toward the service.
    # (Skip when systemd is running us: it sets INVOCATION_ID, so the unit's own
    # `hearth start` stays quiet.)
    if "INVOCATION_ID" not in os.environ and _unit_path().exists():
        print("note: a hearth service is installed — this is a FOREGROUND instance, "
              "not the daemon.\n      For the background service:  hearth service start"
              "      (Ctrl-C stops this one)", file=sys.stderr)

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

    # Refuse to start a second gateway on a port that's already serving — a
    # double-start would orphan the first one (overwritten pidfile) and bind-fail.
    try:
        if httpx.get(f"{_admin_base(cfg)}/admin/health", timeout=1.5).status_code in (200, 503):
            print(f"[hearth] a gateway is already serving on http://{cfg.bind_host}:{cfg.bind_port} "
                  "— use `hearth stop` first to restart", file=sys.stderr)
            return 0
    except httpx.HTTPError:
        pass  # not up — proceed to start

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

    # Hint to run `hearth setup` when the primary_chat model isn't pulled yet.
    pm = cfg.roles.get("primary_chat")
    if ollama and pm:
        try:
            tags = httpx.get(f"{ollama.base_url}/api/tags", timeout=2.0).json().get("models", [])
            names = {t.get("name", "") for t in tags}
            if pm.model not in names:
                print(f"[hearth] primary_chat model '{pm.model}' not pulled — run "
                      f"`hearth setup` to auto-pick + pull a model that fits your hardware "
                      f"(or `hearth pull {pm.model}`)", file=sys.stderr)
        except httpx.HTTPError:
            pass

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
                    return 1
    except httpx.HTTPError as e:
        print(f"\npull failed: {e}", file=sys.stderr)
        return 1
    print("\ndone")
    return 0


def _ensure_ollama(cfg: cfgmod.Config):
    """Return the ollama backend with its daemon up, or None."""
    ollama = cfg.backends.get("ollama")
    if not ollama:
        return None
    if not ollama_supervisor.is_up(ollama.base_url):
        ollama_supervisor.start(ollama.base_url)
        if not ollama_supervisor.is_up(ollama.base_url):
            return None
    return ollama


def cmd_pull(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    ollama = _ensure_ollama(cfg)
    if not ollama:
        print("ollama backend not available", file=sys.stderr)
        return 1
    return _pull_model(ollama.base_url, args.model)


def cmd_setup(args: argparse.Namespace) -> int:
    """First-run wizard: probe hardware → recommend a fitting model set →
    pull + bind. The whole point of `git clone … && hearth setup`."""
    cfg = cfgmod.load()
    rec = hardware.recommend_roles()
    hw = hardware.as_dict()
    ram_gib = (hw.get("ram_total_mib") or 0) // 1024
    print(f"hardware: {hw['inference_target']}  |  RAM {ram_gib} GiB")
    print("recommended role bindings:")
    for r, m in rec.items():
        print(f"  {r:14s} -> {m}")

    if not args.yes:
        try:
            ans = input("\nPull these models and bind the roles? [Y/n] ").strip().lower()
        except EOFError:
            ans = "y"
        if ans in ("n", "no"):
            print("aborted.")
            return 0

    ollama = _ensure_ollama(cfg)
    if not ollama:
        print("ollama backend not available — cannot pull", file=sys.stderr)
        return 1
    for m in sorted(set(rec.values())):
        if _pull_model(ollama.base_url, m) != 0:
            print(f"pull of {m} failed — not binding roles", file=sys.stderr)
            return 1

    for role, model in rec.items():
        cfg.roles[role] = cfgmod.RoleBinding(model=model, backend="ollama")
    cfgmod.save(cfg)
    # Hot-apply to a running gateway so no restart is needed.
    base = _admin_base(cfg)
    for role, model in rec.items():
        try:
            httpx.put(f"{base}/admin/roles/{role}",
                      json={"model": model, "backend": "ollama"}, timeout=3.0)
        except httpx.HTTPError:
            pass
    print("\n✓ setup complete — models pulled, roles bound. Try: hearth chat")
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
        # CLI (e.g. `python -m hearth start`). Find it by the port it serves.
        port = cfgmod.load().bind_port
        found = _pid_on_port(port)
        if found is not None and _pid_looks_like_hearth(found):
            pid, via = found, f"port {port}"
        else:
            print(f"no running hearth gateway found "
                  f"(no pidfile; nothing hearth-like listening on :{port})",
                  file=sys.stderr)
            return 1
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"stopped hearth gateway (pid {pid}, via {via})")
    except ProcessLookupError:
        print("hearth gateway not running (stale reference)")
    except PermissionError:
        print(f"cannot signal pid {pid}: permission denied (owned by another user?)",
              file=sys.stderr)
        return 1
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
    think_on = not bool(getattr(args, "no_think", False))    # deep-think, default on
    show_think = bool(getattr(args, "show_thinking", False))  # show full reasoning text
    bold, dim, reset = "\033[1m", "\033[2m", "\033[0m"
    print(f"{dim}hearth chat — model={model} · deep-think {'on' if think_on else 'off'}\n"
          f"  /exit  /reset  /model NAME  /think on|off  /show  "
          f"(deep-think off = fast lane: fast_chat){reset}")
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


# ---- run as a background service (systemd --user) -----------------------
#
# `hearth` stays the single control surface; underneath it drives a
# systemd --user unit that runs `hearth start`. The unit points at the stable
# ~/.local/bin/hearth symlink — not the in-tree venv — so rebuilding the venv
# or moving the checkout doesn't break the service.

_UNIT_NAME = "hearth.service"
_UNIT_TEMPLATE = """\
[Unit]
Description=hearth — local OpenAI-compatible AI engine (gateway + Ollama)
Documentation=https://github.com/yumima/hearth

[Service]
Type=simple
ExecStart={exec} start
Restart=on-failure
RestartSec=3
TimeoutStopSec=30

[Install]
WantedBy=default.target
"""


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
        print("systemctl not found — `hearth service` needs a systemd user "
              "session. Use `hearth start` directly, or the desktop launcher.",
              file=sys.stderr)
        return 127
    return subprocess.run([sc, "--user", *sc_args], env=_sc_env()).returncode


def _real_hearth_bin() -> str | None:
    """The real `hearth` console-script, or None if there isn't one (e.g. when
    invoked via `python -m hearth`). Never returns __main__.py."""
    found = shutil.which("hearth")
    if found:
        return os.path.realpath(found)
    # The console-script normally sits next to the interpreter in the venv.
    cand = Path(sys.executable).resolve().with_name("hearth")
    if cand.exists():
        return str(cand)
    argv0 = Path(sys.argv[0]).resolve()
    if argv0.name == "hearth" and os.access(argv0, os.X_OK):
        return str(argv0)
    return None


def _launcher() -> str:
    """ExecStart command (without the trailing `start`) for the unit. Prefers
    the stable ~/.local/bin/hearth symlink — (re)pointed at the real console-
    script so venv rebuilds don't break the service — and falls back to
    `<python> -m hearth` so the unit is always a valid, absolute command even
    when no console script is on PATH (never an unexecutable __main__.py)."""
    real = _real_hearth_bin()
    if real:
        link = Path.home() / ".local" / "bin" / "hearth"
        try:
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() and not link.is_symlink():
                return str(link)  # a real file is already installed there; trust it
            if not link.exists() or os.path.realpath(link) != real:
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(real)
            return str(link)
        except OSError:
            return real
    return f"{os.path.realpath(sys.executable)} -m hearth"


def _service_autostart(on: bool, boot: bool) -> int:
    lg = shutil.which("loginctl")
    user = os.environ.get("USER") or str(os.getuid())
    if on:
        rc = _systemctl("enable", _UNIT_NAME)
        if rc == 0:
            print("✓ autostart on login: ENABLED")
            # Only enable boot-linger once login-autostart actually took.
            if boot and lg and subprocess.run([lg, "enable-linger", user]).returncode == 0:
                print("✓ lingering: ENABLED (starts at boot, survives logout)")
        return rc
    rc = _systemctl("disable", _UNIT_NAME)
    if rc == 0:
        print("✓ autostart on login: DISABLED")
    # 'off' fully reverses 'on --boot': drop linger too (no-op if it was off).
    if lg:
        subprocess.run([lg, "disable-linger", user])
    return rc


def _service_install(args: argparse.Namespace) -> int:
    if not shutil.which("systemctl"):
        print("systemctl not found — `hearth service` needs systemd.", file=sys.stderr)
        return 127
    exec_bin = _launcher()
    unit = _unit_path()
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(_UNIT_TEMPLATE.format(exec=exec_bin))
    print(f"✓ installed {unit}")
    print(f"    ExecStart={exec_bin} start")
    _systemctl("daemon-reload")
    if getattr(args, "autostart", False):
        _service_autostart(True, getattr(args, "boot", False))
    print("\nControl with:  hearth service start | stop | restart | status | logs")
    print("Autostart:     hearth service autostart on [--boot]   |   off")
    return 0


def _service_uninstall(args: argparse.Namespace) -> int:
    _systemctl("disable", "--now", _UNIT_NAME)
    _unit_path().unlink(missing_ok=True)
    _systemctl("daemon-reload")
    print("✓ removed hearth.service (stopped if it was running)")
    return 0


def cmd_service(args: argparse.Namespace) -> int:
    action = args.action
    if action == "install":
        return _service_install(args)
    if action == "uninstall":
        return _service_uninstall(args)
    if action == "stop":
        return _systemctl("stop", _UNIT_NAME)
    if action in ("start", "restart"):
        # If a foreground/manual gateway already holds the port, the unit would
        # fork, find the port taken, exit 0, and go inactive — warn rather than
        # silently no-op. (Skip the warning if our service is already active.)
        try:
            port = cfgmod.load().bind_port
        except Exception:
            port = 11435
        active = _systemctl("is-active", "--quiet", _UNIT_NAME) == 0
        if not active and _pid_on_port(port):
            print(f"warning: something already serves :{port} (likely a foreground "
                  f"`hearth start`). Stop it first (`hearth stop`), or the service "
                  f"will start, find the port taken, and immediately exit.",
                  file=sys.stderr)
        return _systemctl(action, _UNIT_NAME)
    if action == "status":
        # `systemctl status` exits non-zero when inactive — don't surface that
        # as a CLI error; the printed status is the answer.
        _systemctl("status", "--no-pager", _UNIT_NAME)
        return 0
    if action == "logs":
        jc = shutil.which("journalctl")
        if not jc:
            print("journalctl not found", file=sys.stderr)
            return 127
        tail = ["-f"] if getattr(args, "follow", False) else ["-n", "200", "--no-pager"]
        return subprocess.run([jc, "--user", "-u", _UNIT_NAME, *tail], env=_sc_env()).returncode
    if action == "autostart":
        return _service_autostart(args.state == "on", getattr(args, "boot", False))
    print(f"unknown service action {action!r}", file=sys.stderr)
    return 2


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

    s = sub.add_parser("setup", help="probe hardware, pull a fitting model, bind roles")
    s.add_argument("--yes", "-y", action="store_true", help="don't prompt; pull + bind")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("status", help="show whether the gateway is up + roles")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("stop", help="stop a running gateway (started via `hearth start`)")
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("chat", help="interactive streaming chat with the local engine")
    s.add_argument("--model", help="role alias or model id (default: primary_chat)")
    s.add_argument("--system", help="optional system prompt")
    s.add_argument("--no-think", action="store_true",
                   help="start with deep-think off (fast lane); toggle in-chat with /think on|off")
    s.add_argument("--show-thinking", action="store_true",
                   help="show the model's full reasoning text (default: compact indicator)")
    s.set_defaults(func=cmd_chat)

    sp = sub.add_parser("service", help="run hearth as a background service (systemd --user)")
    sp_sub = sp.add_subparsers(dest="action", required=True)
    sp_i = sp_sub.add_parser("install", help="install the systemd --user unit")
    sp_i.add_argument("--autostart", action="store_true", help="also enable start-on-login")
    sp_i.add_argument("--boot", action="store_true",
                      help="with --autostart: also start at boot, before login (linger)")
    sp_sub.add_parser("uninstall", help="stop + remove the unit")
    sp_sub.add_parser("start", help="start the service now")
    sp_sub.add_parser("stop", help="stop the service")
    sp_sub.add_parser("restart", help="restart (reloads edited code)")
    sp_sub.add_parser("status", help="show service status")
    sp_l = sp_sub.add_parser("logs", help="show service logs (journal)")
    sp_l.add_argument("-f", "--follow", action="store_true", help="follow live")
    sp_a = sp_sub.add_parser("autostart", help="enable/disable start-on-login")
    sp_a.add_argument("state", choices=["on", "off"])
    sp_a.add_argument("--boot", action="store_true",
                      help="with 'on': also start at boot, before login (linger)")
    sp.set_defaults(func=cmd_service)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
