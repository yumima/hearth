"""CLI command-surface + desktop-launcher tests.

Locks the surface so the recurring app→client→gui→client renames can't silently
break dispatch or the launcher's open-the-window subcommand again. No window is
launched and no real system files are touched — installs are redirected to a
tmp HOME/XDG dir.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from hearth import cli, desktop
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
    ["gui"], ["install"], ["uninstall"], ["app"],     # old flat / app names are gone
    ["client", "url"],                                # the `url` action was dropped
    ["service", "install"], ["service", "uninstall"],  # removed from `service`
])
def test_removed_commands_are_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# ── launcher: every OS opens the window via `<hearth> client open` ─────────────

def test_launch_argv_is_client_open():
    assert desktop._launch_argv(["/x/hearth"]) == ["/x/hearth", "client", "open"]
    assert desktop._launch_argv(["/usr/bin/python", "-m", "hearth"]) == \
        ["/usr/bin/python", "-m", "hearth", "client", "open"]


def test_windows_lnk_target_and_arguments():
    # _install_windows sets TargetPath = argv[0], Arguments = the rest.
    argv = desktop._launch_argv(["C:/h/hearth.exe"])
    assert argv[0] == "C:/h/hearth.exe"
    assert " ".join(argv[1:]) == "client open"


def test_linux_desktop_entry_runs_client_open(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert desktop._install_linux(["/opt/hearth/bin/hearth"]) == 0
    entry = (tmp_path / "applications" / "hearth-chat.desktop").read_text()
    assert "Exec=/opt/hearth/bin/hearth client open" in entry
    assert "Icon=" in entry


def test_macos_app_launcher_runs_client_open(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))            # Path.home() → tmp
    assert desktop._install_macos(["/opt/hearth/bin/hearth"]) == 0
    launcher = (tmp_path / "Applications" / "Hearth.app" / "Contents" / "MacOS"
                / "hearth-chat").read_text()
    assert launcher.strip().endswith("exec /opt/hearth/bin/hearth client open")


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
    monkeypatch.setattr(cli.hardware, "recommend_roles", lambda: {})

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
    monkeypatch.setattr(cli.hardware, "recommend_roles", lambda: {})
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
