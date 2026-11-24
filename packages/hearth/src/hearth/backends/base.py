"""Backend adapter contract (design doc §4, §C3).

Every backend reports capability flags so the gateway can refuse a request
that needs an unsupported capability with a clear error citing the model and
the missing capability — rather than passing it down and getting an opaque
failure from the backend.

v1 ships exactly one adapter (Ollama). llama.cpp / vLLM / LM Studio land as
later adapters behind the same protocol (M4); the route layer never imports a
concrete backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class Capabilities:
    chat: bool = True
    embeddings: bool = False
    streaming: bool = True
    tools: bool = False
    vision: bool = False
    json_mode: bool = False


@dataclass
class ModelInfo:
    id: str
    # extra metadata kept off the OpenAI /v1/models shape; surfaced via
    # /admin/backends instead so OpenAI SDK clients don't choke on it.
    size_bytes: int | None = None
    family: str | None = None


@runtime_checkable
class Backend(Protocol):
    name: str
    base_url: str
    capabilities: Capabilities

    async def health(self) -> bool: ...

    async def list_models(self) -> list[ModelInfo]: ...

    # Chat: non-streaming returns the parsed OpenAI dict; streaming yields raw
    # SSE byte chunks already in OpenAI's `data: {...}\n\n` framing.
    async def chat(self, payload: dict) -> dict: ...

    def chat_stream(self, payload: dict) -> AsyncIterator[bytes]: ...

    async def embeddings(self, payload: dict) -> dict: ...

    # Lifecycle (model manager — design doc §4.5). Pull streams progress lines.
    def pull(self, model: str) -> AsyncIterator[bytes]: ...

    async def delete(self, model: str) -> None: ...

    async def aclose(self) -> None: ...
