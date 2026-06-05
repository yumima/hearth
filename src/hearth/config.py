"""Config + role registry.

Hearth resolves a request's ``model`` field two ways (design doc §4.2):

1. **Role alias** — ``primary_chat`` / ``fast_chat`` / ``coding`` /
   ``embedding`` / ``stt`` / ``vision`` → the role registry → a concrete
   ``(model_id, backend)``.
2. **Literal model id** — ``qwen2.5:14b`` routes straight to the backend
   that has it.

Config lives at ``~/.hearth/config.yaml`` (override via ``HEARTH_CONFIG``).
First run writes the default file so the engine is usable before the user
touches anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Role names finterm (and other consumers) address. Keeping the set closed
# means an OpenAI-SDK client that sends model="primary_chat" always resolves.
ROLE_NAMES = ("primary_chat", "fast_chat", "coding", "embedding", "stt", "vision")

DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_PORT = 11435  # one off from Ollama's 11434 (design doc §4.6)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# Defaults tuned for a 12 GB-VRAM consumer GPU (this box: RTX 5070 Ti 12 GB,
# CUDA compute healthy even though the dGPU is wedged for *display*). Qwen3 is
# the current generation; qwen3:14b (~9 GB q4) fits fully on the GPU. The
# first-run wizard (M3) will re-pick these from the hardware probe.
_DEFAULT_ROLES: dict[str, dict[str, str]] = {
    "primary_chat": {"model": "qwen3:14b", "backend": "ollama"},
    "fast_chat": {"model": "qwen3:8b", "backend": "ollama"},
    "coding": {"model": "qwen3:8b", "backend": "ollama"},
    "embedding": {"model": "nomic-embed-text", "backend": "ollama"},
    # stt/vision left unbound until M2/optional. An unbound role resolves to
    # None and the endpoint returns a clear "role not bound" error.
}


def config_path() -> Path:
    env = os.environ.get("HEARTH_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path(os.environ.get("HEARTH_HOME", "~/.hearth")).expanduser() / "config.yaml"


@dataclass
class Backend:
    name: str
    type: str  # "ollama" | "llamacpp" | "vllm" | "whisper" | ...
    base_url: str


@dataclass
class RoleBinding:
    model: str
    backend: str


@dataclass
class Config:
    bind_host: str = DEFAULT_BIND_HOST
    bind_port: int = DEFAULT_BIND_PORT
    backends: dict[str, Backend] = field(default_factory=dict)
    roles: dict[str, RoleBinding] = field(default_factory=dict)
    path: Path | None = None

    # ---- role resolution -------------------------------------------------
    def resolve(self, model: str) -> tuple[str, Backend]:
        """Map an incoming ``model`` to ``(concrete_model_id, backend)``.

        Role alias first, then literal id (routed to its default backend).
        Raises ``KeyError`` for an unbound role and ``LookupError`` for an
        unknown backend so the route layer can return a precise error.
        """
        binding = self.roles.get(model)
        if binding is not None:
            backend = self.backends.get(binding.backend)
            if backend is None:
                raise LookupError(
                    f"role {model!r} is bound to backend {binding.backend!r} "
                    f"which is not configured"
                )
            return binding.model, backend
        # Literal model id — route to the first ollama-typed backend (M1 has
        # exactly one). Multi-backend literal routing lands with the second
        # adapter (M4).
        backend = self._default_backend()
        return model, backend

    def _default_backend(self) -> Backend:
        if not self.backends:
            raise LookupError("no backends configured")
        # Prefer an ollama backend; otherwise first defined.
        for b in self.backends.values():
            if b.type == "ollama":
                return b
        return next(iter(self.backends.values()))

    def backend_for_role(self, role: str) -> tuple[str, Backend] | None:
        binding = self.roles.get(role)
        if binding is None:
            return None
        backend = self.backends.get(binding.backend)
        if backend is None:
            return None
        return binding.model, backend

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "bind": {"host": self.bind_host, "port": self.bind_port},
            "backends": {
                name: {"type": b.type, "base_url": b.base_url}
                for name, b in self.backends.items()
            },
            "roles": {
                name: {"model": r.model, "backend": r.backend}
                for name, r in self.roles.items()
            },
        }


def default_config() -> Config:
    return Config(
        backends={"ollama": Backend("ollama", "ollama", DEFAULT_OLLAMA_URL)},
        roles={k: RoleBinding(**v) for k, v in _DEFAULT_ROLES.items()},
    )


def load(create: bool = True) -> Config:
    """Load config from disk, writing defaults on first run if ``create``."""
    p = config_path()
    if not p.exists():
        cfg = default_config()
        cfg.path = p
        if create:
            save(cfg)
        return cfg
    raw = yaml.safe_load(p.read_text()) or {}
    bind = raw.get("bind", {})
    backends = {
        name: Backend(name, b.get("type", "ollama"), b["base_url"])
        for name, b in (raw.get("backends") or {}).items()
    }
    if not backends:
        backends = {"ollama": Backend("ollama", "ollama", DEFAULT_OLLAMA_URL)}
    roles = {
        name: RoleBinding(model=r["model"], backend=r.get("backend", "ollama"))
        for name, r in (raw.get("roles") or {}).items()
    }
    return Config(
        bind_host=bind.get("host", DEFAULT_BIND_HOST),
        bind_port=int(bind.get("port", DEFAULT_BIND_PORT)),
        backends=backends,
        roles=roles,
        path=p,
    )


def save(cfg: Config) -> Path:
    p = cfg.path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(cfg.to_yaml_dict(), sort_keys=False))
    tmp.replace(p)  # atomic
    cfg.path = p
    return p
