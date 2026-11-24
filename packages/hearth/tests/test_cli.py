"""CLI command-surface tests.

Locks the surface so the recurring app→client→gui→client renames can't silently
break dispatch or the launcher's open-the-window subcommand again. No window is
launched and no real system files are touched — installs are redirected to a
tmp HOME/XDG dir.
"""

from __future__ import annotations

import socket
import sys

import httpx
import pytest

from hearth import cli
from hearth.cli import build_parser


def _dispatch(argv):
    a = build_parser().parse_args(argv)
    return a.func.__name__, getattr(a, "action", "MISSING")


# ── command dispatch ──────────────────────────────────────────────────────────

def test_bare_hearth_is_terminal_chat():
    assert _dispatch([])[0] == "cmd_chat"


def test_client_subcommands_dispatch():
    assert _dispatch(["client"]) == ("cmd_client", None)            # bare → open
    assert _dispatch(["client", "open"]) == ("cmd_client", "open")
    assert _dispatch(["client", "install"]) == ("cmd_client", "install")
    assert _dispatch(["client", "uninstall"]) == ("cmd_client", "uninstall")


def test_chat_and_service_still_dispatch():
    assert _dispatch(["chat"])[0] == "cmd_chat"
    assert _dispatch(["service", "start"]) == ("cmd_service", "start")


@pytest.mark.parametrize("argv", [
    ["gui"], ["install"], ["app"],     # old flat / app names are gone
    ["client", "url"],                                # the `url` action was dropped
    ["service", "install"], ["service", "uninstall"],  # removed from `service`
])
def test_removed_commands_are_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# ── the chat window is mantel ─────────────────────────────────────────────────

def _client_runs(monkeypatch, program, *argv):
    ran = []
    monkeypatch.setattr(cli, "_mantel", lambda: program)
    started = []
    monkeypatch.setattr(cli, "cmd_start", lambda a: started.append(a.background) or 0)
    monkeypatch.setattr(cli.subprocess, "run", lambda a, **k: ran.append(a) or type("R", (), {"returncode": 7})())
    rc = cli.main(["client", *argv])
    assert started == ([True] if program and argv in ((), ("open",)) else [])  # the window needs the engine
    return rc, ran


def test_client_is_mantel(monkeypatch):
    assert _client_runs(monkeypatch, "/x/mantel") == (7, [["/x/mantel"]])
    assert _client_runs(monkeypatch, "/x/mantel", "open") == (7, [["/x/mantel"]])
    assert _client_runs(monkeypatch, "/x/mantel", "install") == (7, [["/x/mantel", "install"]])
    assert _client_runs(monkeypatch, "/x/mantel", "uninstall") == (7, [["/x/mantel", "uninstall"]])


def test_client_without_mantel_says_so(monkeypatch, capsys):
    assert _client_runs(monkeypatch, None) == (1, [])
    assert "hearth chat" in capsys.readouterr().err


def test_mantel_is_looked_for_where_the_installer_puts_it(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path))
    monkeypatch.setattr(cli.sys, "executable", str(tmp_path / "venv" / "python"))
    monkeypatch.setattr(cli.shutil, "which", lambda n: None)
    assert cli._mantel() is None
    exe = tmp_path / "bin" / ("mantel.exe" if sys.platform == "win32" else "mantel")
    exe.parent.mkdir()
    exe.write_text("")
    assert cli._mantel() == str(exe)


def test_swap_alias_accepts_context_like_bind():
    """`swap` aliases cmd_bind. When --context existed only on `bind`, every
    `hearth swap` died with AttributeError before writing anything."""
    from hearth import cli

    parser = cli.build_parser()
    args = parser.parse_args(["swap", "primary_chat", "qwen3:8b"])
    assert hasattr(args, "context"), "swap must expose --context; cmd_bind reads it"
    assert args.context is None
    assert parser.parse_args(
        ["swap", "primary_chat", "qwen3:8b", "--context", "16384"]).context == 16384


# ── `hearth start` refuses to double-bind ────────────────────────────────────

def _start_args(bind: str, **kw):
    a = build_parser().parse_args(["start", "--bind", bind])
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
        rc = cli.cmd_start(_start_args(f"127.0.0.1:{port}", no_manage=True))

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
        rc = cli.cmd_start(_start_args(f"127.0.0.1:{port}", no_manage=True))

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


def test_start_tip_carries_the_recommended_context(tmp_path, monkeypatch, capsys):
    """Bind clears a role's context when none is given, so a tip without one
    would land the model on the daemon's default and silently drop old messages."""
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    monkeypatch.setattr(cli.catalog, "recommend_roles", lambda hw, here=(): {"primary_chat": "some-model:1b"})
    monkeypatch.setattr(cli.catalog, "recommend_contexts", lambda hw, models: {"primary_chat": 16384})
    monkeypatch.setattr(cli, "_gateway_identity", lambda cfg: None)

    with socket.socket() as squatter:  # stops cmd_start right after the tip
        squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        port = squatter.getsockname()[1]
        cli.cmd_start(_start_args(f"127.0.0.1:{port}", no_manage=True))

    assert "hearth bind primary_chat some-model:1b --context 16384" in capsys.readouterr().err


# ── `hearth bind` ─────────────────────────────────────────────────────────────

def _bind(*argv):
    return cli.cmd_bind(build_parser().parse_args(["bind", *argv]))


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

