"""OpenAI-compatible remote backend adapter.

Fronts any OpenAI-style HTTP API — OpenAI, Zhipu GLM, Together, Groq, OpenRouter,
a remote vLLM, … — behind hearth's role registry, so a cloud model can be bound
to a role and used by every local client (finterm, mantel) through the same
loopback API with no client reconfiguration: just

    hearth backend add zhipu --type openai \\
        --base-url https://open.bigmodel.cn/api/paas/v4 --api-key-env ZHIPU_API_KEY
    hearth bind primary_chat glm-4.6 --backend zhipu

The request is already in OpenAI shape, so this is a near-pure passthrough: we
attach an Authorization header and stream the response bytes unchanged. The
gateway pops hearth's `think` extension before we ever see the payload, so cloud
APIs never receive a non-standard field. Model lifecycle (pull/delete) is not
applicable to a remote API.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from .base import Capabilities, ModelInfo

# Conservative caps for a generic OpenAI-compatible endpoint. Vision/tools vary
# by model; the route only blocks tools when the flag is False, so leaving them
# True keeps capable models working and lets the provider reject the rest.
_CAPS = Capabilities(
    chat=True, embeddings=True, streaming=True, tools=True, vision=True, json_mode=True
)


class OpenAIBackend:
    def __init__(self, name: str, base_url: str, api_key: str = "", timeout: float = 600.0):
        self.name = name
        # base_url is the VERSIONED root (…/v1, …/paas/v4) — endpoints hang
        # directly off it, mirroring how OpenAI SDK base_url works.
        self.base_url = base_url.rstrip("/")
        self.capabilities = _CAPS
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def health(self) -> bool:
        try:
            r = await self._client.get("/models")
            return r.status_code < 500  # 401/403 still means "reachable"
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        try:
            r = await self._client.get("/models")
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError):
            return []  # some providers gate /models behind auth/scope — not fatal
        out: list[ModelInfo] = []
        for m in (data.get("data") or data.get("models") or []):
            mid = (m.get("id") or m.get("name")) if isinstance(m, dict) else str(m)
            if mid:
                out.append(ModelInfo(id=mid, family=self.name))
        return out

    # ---- chat ------------------------------------------------------------
    async def chat(self, payload: dict) -> dict:
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()

    async def chat_stream(self, payload: dict) -> AsyncIterator[bytes]:
        body = {**payload, "stream": True}
        async with self._client.stream("POST", "/chat/completions", json=body) as r:
            r.raise_for_status()
            async for chunk in r.aiter_raw():
                if chunk:
                    yield chunk

    async def embeddings(self, payload: dict) -> dict:
        r = await self._client.post("/embeddings", json=payload)
        r.raise_for_status()
        return r.json()

    # ---- lifecycle (N/A for a remote API) --------------------------------
    async def pull(self, model: str) -> AsyncIterator[bytes]:
        yield (b'{"status":"error: a remote (openai) backend cannot pull models; '
               b'the provider hosts them"}\n')

    async def delete(self, model: str) -> None:
        raise NotImplementedError("delete is not supported for a remote (openai) backend")

    async def aclose(self) -> None:
        await self._client.aclose()
