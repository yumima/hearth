"""CLI command-surface tests.

Locks the surface so the recurring app→client→gui→client renames can't silently
break dispatch or the launcher's open-the-window subcommand again. No window is
launched and no real system files are touched — installs are redirected to a
tmp HOME/XDG dir.
"""

from __future__ import annotations

import re
import socket

import httpx
import pytest

from hearth import cli
from hearth.cli import build_parser


def _dispatch(argv):
    a = build_parser().parse_args(argv)
    return a.func.__name__, getattr(a, "action", "MISSING")


# ── command dispatch ──────────────────────────────────────────────────────────

def test_bare_hearth_is_the_coding_agent():
    assert _dispatch([])[0] == "cmd_code"
    a = build_parser().parse_args(["-p", "do it", "-q"])
    assert (a.func.__name__, a.prompt, a.quiet, a.path, a.yolo) == ("cmd_code", "do it", True, ".", False)
    assert build_parser().parse_args(["code", "sub", "-p", "x"]).path == "sub"


@pytest.mark.parametrize("argv,func", [
    (["chat"], "cmd_chat"), (["code", "x"], "cmd_code"), (["client"], "cmd_client"), (["search", "x"], "cmd_search"),
    (["start"], "cmd_start"), (["serve"], "cmd_serve"), (["stop"], "cmd_stop"), (["restart"], "cmd_restart"),
    (["status"], "cmd_status"), (["logs", "-f"], "cmd_logs"), (["autostart"], "cmd_autostart"), (["autostart", "on"], "cmd_autostart"),
    (["model", "list"], "cmd_model_list"), (["model", "suggest"], "cmd_model_suggest"), (["model", "install", "m"], "cmd_model_install"),
    (["model", "add", "m"], "cmd_model_install"), (["model", "use", "m"], "cmd_model_use"), (["model", "remove", "m"], "cmd_model_remove"),
    (["voice", "list"], "cmd_voice_list"), (["voice", "install", "kokoro"], "cmd_voice_install"), (["voice", "remove", "x"], "cmd_voice_remove"),
    (["provider", "list"], "cmd_provider"), (["provider", "add", "n", "--base-url", "u"], "cmd_provider"), (["provider", "remove", "n"], "cmd_provider"),
    (["setup"], "cmd_setup"), (["update"], "cmd_update"), (["uninstall"], "cmd_uninstall"),
    (["start", "--background"], "cmd_start"),  # as apps say it
])
def test_every_command(argv, func):
    assert _dispatch(argv)[0] == func


def test_the_help_names_every_command_and_no_other():
    import argparse

    p = build_parser()
    subs = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction)).choices
    text = cli._HELP
    for name, sp in subs.items():
        assert re.search(rf"\b{name}\b", text), name
        for g in (a for a in sp._actions if isinstance(a, argparse._SubParsersAction)):
            line = next(ln for ln in text.splitlines() if ln.strip().startswith(f"hearth {name} "))
            assert set(re.findall(r"[a-z]+", line.split("…", 1)[1].split(":")[0])) == set(g.choices) - {"add"} | (
                {"add"} if name == "provider" else set())
    named = set(re.findall(r"^  hearth ([a-z]+)", text, re.M)) | set(re.findall(r"· ([a-z]+)", "\n".join(
        ln for ln in text.splitlines() if re.match(r"  hearth (start|update)", ln))))
    assert named <= set(subs)


def test_one_name_for_one_thing():
    for old in ("models", "pull", "bind", "swap", "roles", "plan", "hardware", "install", "remove", "backend", "service"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([old])


def test_a_group_without_a_command_says_which_there_are(capsys):
    for g in ("model", "voice", "provider"):
        with pytest.raises(SystemExit):
            cli.main([g])
        assert f"hearth {g}" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["gui"], ["install"], ["app"],     # old flat / app names are gone
    ["client", "url"],                                # the `url` action was dropped
    ["service", "install"], ["service", "uninstall"],  # removed from `service`
])
def test_removed_commands_are_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# ── `hearth serve` refuses to double-bind ────────────────────────────────────

def _start_args(bind: str, **kw):
    a = build_parser().parse_args(["serve", "--bind", bind])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_duplicate_start_survives_a_slow_health_probe(tmp_path, monkeypatch, capsys):
    """A gateway that is up but slow to answer must still be detected.

    /admin/health awaits every configured backend — remote providers included —
    so one slow round-trip over the internet timed the old probe out, and the
    second start concluded the port was free, printed "serving ...", and then
    died on EADDRINUSE. The probe is /admin/version, a constant handler.
    """
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    monkeypatch.setattr(cli.catalog, "recommend_roles", lambda hw, here=(): {})

    def fake_get(url, **kw):
        if "/admin/health" in url:
            raise httpx.ReadTimeout("backend round-trip is slow", request=None)
        if "/admin/version" in url:
            return httpx.Response(200, headers={"server": "hearth/9.9.9"},
                                  json={"name": "hearth", "version": "9.9.9"})
        return httpx.Response(200, json={"models": []})  # e.g. ollama /api/tags

    monkeypatch.setattr(cli.httpx, "get", fake_get)

    # A real listener on the port, so a regressed guard fails the bind fast
    # instead of serving forever and hanging the suite.
    with socket.socket() as served:
        served.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        served.bind(("127.0.0.1", 0))
        served.listen(1)
        port = served.getsockname()[1]
        rc = cli.cmd_serve(_start_args(f"127.0.0.1:{port}", no_manage=True))

    out, err = capsys.readouterr()
    assert rc == 0
    assert "already serving" in err
    assert "serving OpenAI-compatible" not in out
    assert not (tmp_path / "hearth.pid").exists()


def test_start_on_a_taken_port_touches_nothing(tmp_path, monkeypatch, capsys):
    """A port held by someone else is reported before anything is announced:
    no "serving" line and no pidfile — a doomed start used to write one and then
    delete the live gateway's pidfile on its way out."""
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    monkeypatch.setattr(cli.catalog, "recommend_roles", lambda hw, here=(): {})
    monkeypatch.setattr(cli, "_gateway_identity", lambda cfg: None)  # occupant isn't hearth

    with socket.socket() as squatter:
        squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        port = squatter.getsockname()[1]
        rc = cli.cmd_serve(_start_args(f"127.0.0.1:{port}", no_manage=True))

    out, err = capsys.readouterr()
    assert rc == 1
    assert "already in use" in err
    assert "serving OpenAI-compatible" not in out
    assert not (tmp_path / "hearth.pid").exists()


def test_listen_socket_is_listening_and_ipv4_for_hostnames():
    """Bound AND listening, so a racing second start's bind fails at once; and a
    hostname binds IPv4 (uvicorn's rule), not whatever getaddrinfo lists first."""
    sock = cli._listen_socket("localhost", 0)
    try:
        assert sock.family == socket.AF_INET
        port = sock.getsockname()[1]
        with pytest.raises(OSError):
            cli._listen_socket("127.0.0.1", port)
    finally:
        sock.close()


def test_the_tip_is_to_install_it(tmp_path, monkeypatch, capsys):
    """Installing measures the context that fits; a model only written down would
    get the daemon's default and silently drop old messages."""
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    monkeypatch.setattr(cli.catalog, "recommend_roles", lambda hw, here=(): {"primary_chat": "some-model:1b"})
    monkeypatch.setattr(cli, "_gateway_identity", lambda cfg: None)

    with socket.socket() as squatter:  # stops cmd_start right after the tip
        squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        port = squatter.getsockname()[1]
        cli.cmd_serve(_start_args(f"127.0.0.1:{port}", no_manage=True))

    assert "hearth model install some-model:1b" in capsys.readouterr().err


# ── `hearth model use --for` ─────────────────────────────────────────────────────────────

def _bind(role, model, *argv):
    return cli.main(["model", "use", model, "--for", role, *[{"--backend": "--provider"}.get(a, a) for a in argv]])


def _gateway_says(monkeypatch, status):
    monkeypatch.setattr(cli.httpx, "put", lambda url, **kw: httpx.Response(
        status, request=httpx.Request("PUT", url)))


def test_bind_reports_when_the_gateway_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    _gateway_says(monkeypatch, 401)
    assert _bind("primary_chat", "some-model") == 0
    out, err = capsys.readouterr()
    assert "applied to running gateway" not in out
    assert "HTTP 401" in err


def test_bind_reports_when_the_gateway_applies(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    _gateway_says(monkeypatch, 200)
    _bind("primary_chat", "some-model")
    assert "applied to running gateway" in capsys.readouterr().out


def test_bind_rejects_a_negative_context_without_saving(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    before = cli.cfgmod.load().roles["primary_chat"].model
    assert _bind("primary_chat", "some-model", "--context", "-1") == 2
    assert cli.cfgmod.load().roles["primary_chat"].model == before


def test_bind_matches_the_context_of_other_roles_on_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    _gateway_says(monkeypatch, 200)
    _bind("primary_chat", "some-model", "--context", "32768")
    _bind("fast_chat", "some-model")
    assert cli.cfgmod.load().roles["fast_chat"].context == 32768


def test_bind_explicit_zero_context_still_clears(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    _gateway_says(monkeypatch, 200)
    _bind("primary_chat", "some-model", "--context", "32768")
    _bind("fast_chat", "some-model", "--context", "0")
    assert cli.cfgmod.load().roles["fast_chat"].context == 0


def test_pull_refused_by_old_daemon_names_the_version(monkeypatch, capsys):
    class Refused:
        status_code = 412

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_lines(self):
            return iter(())

        def read(self):
            return b""

        def json(self):
            return {"error": "requires a newer version of Ollama"}

    monkeypatch.setattr(cli.httpx, "stream", lambda *a, **kw: Refused())
    monkeypatch.setattr(cli.httpx, "get", lambda url, **kw: httpx.Response(200, json={"version": "0.30.4"}))
    assert cli._pull_model("http://127.0.0.1:1", "new-model") == 1
    err = capsys.readouterr().err
    assert "requires a newer version" in err and "Ollama 0.30.4" in err



# ── `hearth model use`, `voice`, `restart` ────────────────────────────────────

@pytest.fixture
def conf(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTH_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setattr(cli.httpx, "put", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("none runs")))
    jobs = []
    monkeypatch.setattr(cli, "_run_job", lambda *a: jobs.append(a) or 0)
    return type("C", (), {"jobs": jobs, "load": staticmethod(cli.cfgmod.load)})()


def test_a_local_model_is_used_as_it_is_installed(conf):
    """Measured and checked: not only written down."""
    assert cli.main(["model", "use", "m:1b"]) == 0 and cli.main(["model", "use", "m:1b", "--anyway"]) == 0
    assert conf.jobs == [("model", "m:1b", False), ("model", "m:1b", True)]


def test_a_providers_model_answers_for_chat_and_agent(conf):
    assert cli.main(["provider", "add", "cloud", "--base-url", "https://c.example/v1", "--api-key-env", "K"]) == 0
    assert cli.main(["model", "use", "big", "--provider", "cloud"]) == 0 and conf.jobs == []
    roles = conf.load().roles
    assert {r: (roles[r].model, roles[r].backend) for r in ("primary_chat", "fast_chat", "coding")} == {
        r: ("big", "cloud") for r in ("primary_chat", "fast_chat", "coding")}
    assert cli.main(["model", "use", "eye", "--provider", "cloud", "--for", "vision", "--for", "stt"]) == 0
    assert conf.load().roles["vision"].model == conf.load().roles["stt"].model == "eye"
    assert conf.load().roles["coding"].model == "big"


def test_what_is_not_there_is_said(conf, capsys):
    assert cli.main(["model", "use", "x", "--provider", "nowhere"]) == 2 and "provider list" in capsys.readouterr().err
    assert cli.main(["model", "use", "x", "--context", "-1"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["model", "use", "x", "--for", "no_such_role"])


def test_voices_come_and_go(conf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "data"))
    d = cli.paths.voices_dir("piper")
    d.mkdir(parents=True)
    for n in ("a.onnx", "a.onnx.json", "ab.onnx"):
        (d / n).write_text("x")
    assert cli.main(["voice", "list"]) == 0 and "ab " in capsys.readouterr().out
    assert cli.main(["voice", "remove", "a"]) == 0 and [p.name for p in d.iterdir()] == ["ab.onnx"]
    assert cli.main(["voice", "remove", "a"]) == 1
    for bad in ("../ab", "..", "/etc/passwd", ".hidden", ""):
        assert cli.main(["voice", "remove", bad]) == 2
    assert (d / "ab.onnx").exists()
    assert cli.main(["voice", "install", "kokoro"]) == 0 and conf.jobs == [("kokoro",)]


@pytest.fixture
def engine(monkeypatch):
    """No real process or service: what would be done to them."""
    did, unit, pid = [], {"exists": False, "state": "inactive"}, [None]
    monkeypatch.setattr(cli, "_has_service", lambda: unit["exists"])
    monkeypatch.setattr(cli, "_unit_is", lambda state: unit["state"] == state)
    monkeypatch.setattr(cli, "_systemctl", lambda *a: did.append(a[0]) or 0)
    monkeypatch.setattr(cli, "_engine_pid", lambda: (pid[0], "pidfile"))
    monkeypatch.setattr(cli, "_gateway_identity", lambda cfg: None)
    from hearth import background

    def start(cfg, argv, service=None):
        if service is not None:
            service.start()
        else:
            did.append(argv[argv.index("hearth") + 1:])
        return True, "running"

    monkeypatch.setattr(background, "start", start)
    monkeypatch.setattr(cli, "_ask_to_stop", lambda cfg, p: did.append(f"end {p}") or True)
    monkeypatch.setattr(cli.procs, "alive", lambda p: False)
    return type("E", (), {"did": did, "unit": unit, "pid": pid})()


def test_where_there_is_a_service_it_is_the_service(engine):
    engine.unit.update(exists=True, state="active")
    assert cli.main(["stop"]) == 0 and cli.main(["restart"]) == 0 and cli.main(["start"]) == 0
    assert engine.did == ["stop", "restart", "start"]
    engine.did.clear()
    engine.unit["state"] = "activating"  # it ends and is started again, over and over
    assert cli.main(["stop"]) == 0 and engine.did == ["stop"]
    engine.did.clear()
    engine.unit["state"], engine.pid[0] = "inactive", 7  # one was started beside it
    assert cli.main(["restart"]) == 0 and engine.did == ["end 7", "restart"]


def test_restart_without_a_service(engine, monkeypatch):
    assert cli.main(["restart"]) == 0 and engine.did == [["serve"]]
    engine.did.clear()
    engine.pid[0] = 7  # answering or not
    monkeypatch.setattr(cli.procs, "argv", lambda p: ["python", "-P", "-m", "hearth", "serve", "--bind", "127.0.0.1:5000", "--no-manage"])
    assert cli.main(["restart"]) == 0 and engine.did == ["end 7", ["serve", "--bind", "127.0.0.1:5000", "--no-manage"]]
    engine.did.clear()
    monkeypatch.setattr(cli.procs, "argv", lambda p: ["hearth", "serve", "--unsafe-bind", "--api-key", "k"])
    assert cli.main(["restart"]) == 2 and engine.did == []  # what cannot be started again is not stopped
    monkeypatch.setattr(cli.procs, "alive", lambda p: True)
    monkeypatch.setattr(cli.procs, "argv", lambda p: ["hearth", "serve"])
    assert cli.main(["restart"]) == 1 and [d for d in engine.did if isinstance(d, list)] == []  # not a second one beside it


def test_a_service_may_have_an_address_others_can_reach(engine, conf, monkeypatch, capsys):
    cfg = conf.load()
    cfg.bind_host = "0.0.0.0"
    monkeypatch.setattr(cli.cfgmod, "load", lambda *a, **k: cfg)
    assert cli.main(["start"]) == 2 and "unsafe-bind" in capsys.readouterr().err
    engine.unit.update(exists=True, state="active")
    assert cli.main(["start"]) == 0 and cli.main(["restart"]) == 0 and engine.did == ["start", "restart"]


def test_the_services_log_also_when_it_failed(engine, monkeypatch):
    ran = []
    monkeypatch.setattr(cli.shutil, "which", lambda n: f"/usr/bin/{n}")
    monkeypatch.setattr(cli, "_sc_env", dict)
    monkeypatch.setattr(cli.subprocess, "run", lambda a, **k: ran.append(a[0]) or type("R", (), {"returncode": 0})())
    engine.unit.update(exists=True, state="failed")
    assert cli.main(["logs"]) == 0 and ran == ["/usr/bin/journalctl"]
    engine.unit["exists"] = False
    assert cli.main(["logs"]) == 0 and len(ran) == 1


def test_the_models_here_and_what_they_are_for(conf, monkeypatch, capsys):
    cfg = conf.load()
    cfg.roles["primary_chat"] = cli.cfgmod.RoleBinding("m", "ollama")
    cfg.roles["coding"] = cli.cfgmod.RoleBinding("gone:1b", "ollama")
    cli.cfgmod.save(cfg)
    monkeypatch.setattr(cli, "_ensure_ollama", lambda c: c.backends["ollama"])
    monkeypatch.setattr(cli.httpx, "get", lambda *a, **k: httpx.Response(200, json={"models": [
        {"name": "m:latest", "size": 4_000_000_000}, {"name": "spare:2b", "size": 1_000_000_000}]}))
    assert cli.main(["model", "list"]) == 0
    out = {ln.split()[0]: ln for ln in capsys.readouterr().out.splitlines() if ln.strip()}
    assert "4.0 GB" in out["m:latest"] and "primary_chat" in out["m:latest"]
    assert out["spare:2b"].rstrip().endswith("—") and "not here" in out["gone:1b"] and "coding" in out["gone:1b"]


def test_a_model_is_not_shown_as_here_because_one_of_its_name_is(conf, monkeypatch, capsys):
    cfg = conf.load()
    cfg.backends["groq"] = cli.cfgmod.Backend("groq", "openai", "https://g.example/v1", "", "K")
    cfg.backends["vllm"] = cli.cfgmod.Backend("vllm", "openai", "http://127.0.0.1:8000/v1", "k", "")
    for role, backend in (("primary_chat", "groq"), ("coding", "ollama"), ("vision", "vllm")):
        cfg.roles[role] = cli.cfgmod.RoleBinding("m:1b", backend)
    cli.cfgmod.save(cfg)
    monkeypatch.setattr(cli, "_ensure_ollama", lambda c: c.backends["ollama"])
    monkeypatch.setattr(cli.httpx, "get", lambda *a, **k: httpx.Response(200, json={"models": [{"name": "m:1b", "size": 1}]}))
    assert cli.main(["model", "list"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("m:1b")]
    assert ["GB" in ln and "coding" in ln and "primary_chat" not in ln for ln in lines] == [True, False, False]
    assert any("⚠ groq" in ln and "primary_chat" in ln for ln in lines)
    assert any("vllm" in ln and "⚠" not in ln and "vision" in ln for ln in lines)
