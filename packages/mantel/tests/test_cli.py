"""CLI command-surface dispatch + config save/load roundtrip."""

from __future__ import annotations

import pytest

from mantel import config as cfgmod
from mantel.cli import build_parser


def _d(argv):
    a = build_parser().parse_args(argv)
    return a.func.__name__, getattr(a, "action", None)


def test_bare_opens_window():
    assert _d([])[0] == "cmd_open"


def test_command_dispatch():
    assert _d(["setup"])[0] == "cmd_setup"
    assert _d(["status"])[0] == "cmd_status"
    assert _d(["doctor"])[0] == "cmd_doctor"
    assert _d(["models"])[0] == "cmd_models"
    assert _d(["serve"])[0] == "cmd_serve"
    assert _d(["config"]) == ("cmd_config", "show")
    assert _d(["config", "edit"]) == ("cmd_config", "edit")
    assert _d(["mcp", "list"]) == ("cmd_mcp", "list")
    assert _d(["mcp", "enable", "fs"]) == ("cmd_mcp", "enable")
    assert _d(["provider", "use", "openai"]) == ("cmd_provider", "use")
    assert _d(["install"])[0] == "cmd_install"
    assert _d(["uninstall"])[0] == "cmd_uninstall"


@pytest.mark.parametrize("argv", [["bogus"], ["mcp", "bogus"], ["mcp"], ["provider"]])
def test_invalid_commands_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from mantel.mcp_host import ServerConfig

    cfg = cfgmod.Config()
    cfg.provider = "hearth"
    cfg.model = "qwen3:8b"
    cfg.mcp_servers = {"fs": ServerConfig(command="npx", args=["-y", "srv", "/home"],
                                          enabled=True, auto_approve=["read_*"])}
    path = cfgmod.save(cfg)
    assert path.exists()

    loaded = cfgmod.load()
    assert loaded.provider == "hearth" and loaded.model == "qwen3:8b"
    s = loaded.mcp_servers["fs"]
    assert s.command == "npx" and s.enabled is True and s.auto_approve == ["read_*"]
