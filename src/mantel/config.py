"""Mantel configuration — providers + the active model.

M0 scope: a single default provider (local **hearth**, OpenAI-compatible on
loopback) and the active model. Stored at ``~/.config/mantel/config.yaml``;
sensible defaults apply when it's absent, so a fresh install Just Works against
a running hearth with zero config.

Forward shape (M1+): multiple providers (OpenAI/Anthropic/…), profiles that
bundle ``provider + model + mcp servers + system prompt``, and an ``mcp_servers``
section. Kept deliberately small here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# hearth's OpenAI-compatible gateway on loopback (see github.com/yumima/hearth).
DEFAULT_HEARTH_URL = "http://127.0.0.1:11435/v1"


@dataclass
class Provider:
    """An OpenAI-compatible backend. ``api_key`` is empty for local hearth."""
    type: str = "openai"          # only "openai" (compatible) in M0
    base_url: str = DEFAULT_HEARTH_URL
    api_key: str = ""

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


@dataclass
class Config:
    host: str = "127.0.0.1"       # mantel's own local server (loopback only)
    port: int = 8765
    provider: str = "hearth"      # the active provider key
    model: str = "primary_chat"   # hearth role alias by default
    vision_model: str = "vision"  # model/role used when a message carries an image
                                  # (hearth's `vision` role); "" disables routing
    tts_voice: str = ""           # voice id for read-aloud (e.g. "en_US-amy-medium");
                                  # "" lets the backend pick its default voice
    web_search: bool = True       # ask hearth to add its web search/fetch to chats;
                                  # false keeps a chat (and your tools' results) off the web
    providers: dict[str, Provider] = field(default_factory=lambda: {"hearth": Provider()})
    # name -> mcp_host.ServerConfig (kept as Any to avoid importing the MCP SDK
    # at config-import time; empty by default → mantel behaves as a plain chat).
    mcp_servers: dict = field(default_factory=dict)

    def active(self) -> Provider:
        return self.providers.get(self.provider, Provider())


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "mantel" / "config.yaml"


def state_dir() -> Path:
    """Per-user state dir for runtime files (pidfile, server log). XDG default
    is ~/.local/state/mantel — separate from config so it can be wiped safely."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "mantel"


def pid_path() -> Path:
    return state_dir() / "mantel.pid"


def log_path() -> Path:
    return state_dir() / "mantel.log"


def load() -> Config:
    """Load config, falling back to all-defaults (hearth on :11435) when the file
    is absent or unreadable — mantel must never fail to start over config."""
    cfg = Config()
    p = config_path()
    if not p.exists():
        return cfg
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return cfg
    cfg.host = data.get("host", cfg.host)
    try:
        cfg.port = int(data.get("port", cfg.port))
    except (TypeError, ValueError):
        pass
    cfg.provider = data.get("provider", cfg.provider)
    cfg.model = data.get("model", cfg.model)
    cfg.vision_model = data.get("vision_model", cfg.vision_model)
    cfg.tts_voice = data.get("tts_voice", cfg.tts_voice)
    # Off unless it plainly says on: a typo must not put a chat on the web.
    cfg.web_search = data.get("web_search", cfg.web_search) is True
    provs = data.get("providers") or {}
    if isinstance(provs, dict) and provs:
        cfg.providers = {}
        for name, v in provs.items():
            v = v or {}
            cfg.providers[name] = Provider(
                type=v.get("type", "openai"),
                base_url=v.get("base_url", DEFAULT_HEARTH_URL),
                api_key=v.get("api_key", ""),
            )
    cfg.providers.setdefault(cfg.provider, Provider())  # active must resolve

    mcps = data.get("mcp_servers") or {}
    if isinstance(mcps, dict) and mcps:
        from .mcp_host import ServerConfig  # lazy — only import the MCP SDK if configured
        out: dict = {}
        for name, v in mcps.items():
            v = v or {}
            out[name] = ServerConfig(
                command=v.get("command", ""),
                args=list(v.get("args") or []),
                env=dict(v.get("env") or {}),
                enabled=bool(v.get("enabled", False)),
                auto_approve=list(v.get("auto_approve") or []),
            )
        cfg.mcp_servers = out
    return cfg


def save(cfg: Config) -> Path:
    """Write ``cfg`` back to ~/.config/mantel/config.yaml; returns the path."""
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "host": cfg.host,
        "port": cfg.port,
        "provider": cfg.provider,
        "model": cfg.model,
        "vision_model": cfg.vision_model,
        "tts_voice": cfg.tts_voice,
        "web_search": cfg.web_search,
        "providers": {
            n: {"type": pr.type, "base_url": pr.base_url,
                **({"api_key": pr.api_key} if pr.api_key else {})}
            for n, pr in cfg.providers.items()
        },
    }
    if cfg.mcp_servers:
        data["mcp_servers"] = {
            n: {"command": s.command, "args": s.args, "env": s.env,
                "enabled": s.enabled, "auto_approve": s.auto_approve}
            for n, s in cfg.mcp_servers.items()
        }
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    try:
        p.chmod(0o600)  # may hold provider API keys → owner-only
    except OSError:
        pass
    return p
