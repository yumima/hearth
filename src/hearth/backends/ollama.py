"""Ollama backend adapter.

Ollama already serves an OpenAI-compatible surface (`/v1/chat/completions`,
`/v1/embeddings`, `/v1/models`) plus a native lifecycle API (`/api/pull`,
`/api/delete`, `/api/tags`, `/api/version`). Hearth fronts it to add the
role registry, unified admin, host-header security, and (M2) tool-call
repair — so this adapter is mostly a faithful streaming proxy.

The single standalone Ollama binary bundles its own CUDA runtime, so on this
box it lights up the RTX 5070 Ti (Blackwell / CUDA 13) without a system CUDA
toolkit. We never touch the GL/Vulkan display path the dGPU is wedged on.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .base import Capabilities, ModelInfo

# Ollama serves both chat and embeddings, streams, and (for tool-capable
# models like Qwen2.5) emits OpenAI `tool_calls`.
_CAPS = Capabilities(
    chat=True, embeddings=True, streaming=True, tools=True, vision=True, json_mode=True
)


class OllamaBackend:
    def __init__(self, name: str, base_url: str, timeout: float = 600.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.capabilities = _CAPS
        # Long read timeout: local generation of a long answer can exceed the
        # httpx default. Connect timeout stays short so a dead backend fails
        # fast into the consumer's no-failover banner.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def health(self) -> bool:
        try:
            r = await self._client.get("/api/version")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        data = r.json()
        out: list[ModelInfo] = []
        for m in data.get("models", []):
            details = m.get("details") or {}
            out.append(
                ModelInfo(
                    id=m.get("name") or m.get("model", ""),
                    size_bytes=m.get("size"),
                    family=details.get("family"),
                )
            )
        return out

    # ---- chat ------------------------------------------------------------
    async def chat(self, payload: dict) -> dict:
        r = await self._client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()

    async def chat_stream(self, payload: dict) -> AsyncIterator[bytes]:
        body = {**payload, "stream": True}
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=body
        ) as r:
            r.raise_for_status()
            async for chunk in r.aiter_raw():
                if chunk:
                    yield chunk

    # ---- embeddings ------------------------------------------------------
    async def embeddings(self, payload: dict) -> dict:
        r = await self._client.post("/v1/embeddings", json=payload)
        r.raise_for_status()
        return r.json()

    # ---- lifecycle -------------------------------------------------------
    async def pull(self, model: str) -> AsyncIterator[bytes]:
        body = {"model": model, "stream": True}
        async with self._client.stream("POST", "/api/pull", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.strip():
                    yield (line + "\n").encode()

    async def delete(self, model: str) -> None:
        r = await self._client.request(
            "DELETE", "/api/delete", json={"model": model}
        )
        r.raise_for_status()

    async def ps(self) -> list[dict]:
        r = await self._client.get("/api/ps")
        r.raise_for_status()
        return r.json().get("models", [])

    async def aclose(self) -> None:
        await self._client.aclose()

    # Convenience for the CLI's blocking pull (prints progress nicely).
    @staticmethod
    def parse_progress(line: bytes) -> str | None:
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        status = obj.get("status")
        if not status:
            return None
        total, completed = obj.get("total"), obj.get("completed")
        if total and completed:
            pct = 100.0 * completed / total
            return f"{status} {pct:5.1f}% ({completed/1e9:.2f}/{total/1e9:.2f} GB)"
        return status
