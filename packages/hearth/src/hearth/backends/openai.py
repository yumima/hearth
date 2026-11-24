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

import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable, TypeVar

import httpx

from .base import Capabilities, ModelInfo

# Hosted providers shed load with a 503 rather than queueing — measured against
# Gemini, where roughly one call in three returned 503 under normal use and the
# immediate retry succeeded. Without this the gateway turns a blip the provider
# expects us to retry into a hard 502 at the client. (Quota 429s are a different
# animal and are handled separately — see _retry_delay.)
_RETRIES = 2
_BACKOFF_S = 0.6
# How long a remote provider's model catalogue stays fresh. Catalogues change
# on the order of weeks; consumers poll /v1/models on the order of seconds.
_MODELS_TTL_S = 300.0
# Reachability changes fast; the catalogue does not. Separate, shorter TTL.
_HEALTH_TTL_S = 15.0
# 429 is deliberately NOT here. A capacity blip (503) clears in milliseconds;
# a rate/quota limit does not, and retrying on our own schedule spends more of
# the exact budget that just ran out — turning one failed call into three.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 500, 502, 503, 504})
# We honour a 429 only when the provider says how long to wait, and only if the
# wait is short enough that a caller is still there to receive the answer.
_RETRY_AFTER_CAP_S = 10.0

T = TypeVar("T")


def _retry_delay(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying ``exc``, or None to give up now."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            raw = exc.response.headers.get("retry-after", "")
            try:
                delay = float(raw)
            except ValueError:
                return None  # no usable hint (Google sends none) — fail fast
            return delay if 0 <= delay <= _RETRY_AFTER_CAP_S else None
        if code not in _TRANSIENT_STATUS:
            return None
    # Connect/read timeouts and dropped connections are worth one more try;
    # anything else (bad TLS, invalid URL) is not.
    elif not isinstance(exc, (httpx.TimeoutException, httpx.ConnectError,
                              httpx.ReadError, httpx.RemoteProtocolError)):
        return None
    return _BACKOFF_S * (2 ** attempt)

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
        # (fetched_at_monotonic, models) — see list_models().
        self._models_cache: tuple[float, list[ModelInfo]] | None = None
        # (checked_at_monotonic, healthy) — see health().
        self._health_cache: tuple[float, bool] | None = None

    async def health(self) -> bool:
        """Reachability, cached briefly.

        /admin/health probes every backend, so an unauthenticated consumer
        polling it would bill an outbound provider call per poll — the same
        trap as /v1/models. The TTL is short enough that a real outage still
        shows up promptly, long enough that polling can't turn into spend.
        """
        now = time.monotonic()
        if self._health_cache is not None:
            checked_at, healthy = self._health_cache
            if now - checked_at < _HEALTH_TTL_S:
                return healthy
        try:
            r = await self._client.get("/models")
            healthy = r.status_code < 500  # 401/403 still means "reachable"
        except httpx.HTTPError:
            healthy = False
        self._health_cache = (now, healthy)
        return healthy

    async def list_models(self) -> list[ModelInfo]:
        """Remote catalogue, cached — GET /v1/models fans out to every backend.

        A consumer polling the gateway's model list (mantel does, every 5s)
        would otherwise turn each poll into an outbound provider call: ~17k
        Google requests/day from an idle app, against a free tier of ~1.5k
        requests/day. A provider's catalogue changes on the order of weeks, so
        a short TTL costs nothing and removes the entire class of problem.
        """
        now = time.monotonic()
        if self._models_cache is not None:
            cached_at, cached = self._models_cache
            if now - cached_at < _MODELS_TTL_S:
                return cached
        try:
            r = await self._client.get("/models")
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError):
            # Serve a stale catalogue over none: a blip shouldn't empty a
            # consumer's model dropdown. Some providers also gate /models
            # behind auth/scope entirely — not fatal either way.
            if self._models_cache is not None:
                return self._models_cache[1]
            return []
        out: list[ModelInfo] = []
        for m in (data.get("data") or data.get("models") or []):
            mid = (m.get("id") or m.get("name")) if isinstance(m, dict) else str(m)
            if mid:
                out.append(ModelInfo(id=mid, family=self.name))
        self._models_cache = (now, out)
        return out

    async def _with_retry(self, call: Callable[[], Awaitable[T]]) -> T:
        """Run ``call``, retrying the errors a hosted provider expects us to."""
        for attempt in range(_RETRIES + 1):
            try:
                return await call()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                delay = _retry_delay(exc, attempt)
                if attempt >= _RETRIES or delay is None:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    # ---- chat ------------------------------------------------------------
    async def chat(self, payload: dict) -> dict:
        async def _once() -> dict:
            r = await self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()

        return await self._with_retry(_once)

    async def chat_stream(self, payload: dict) -> AsyncIterator[bytes]:
        body = {**payload, "stream": True}
        # Retry only while nothing has reached the client. raise_for_status()
        # fires before the first chunk, but aiter_raw() can raise ReadTimeout /
        # ReadError / RemoteProtocolError MID-stream — and re-issuing the POST
        # then restarts from byte 0, duplicating the opening of the answer the
        # user already saw. Track it explicitly rather than reasoning about
        # which exception types can occur where.
        for attempt in range(_RETRIES + 1):
            yielded_any = False
            try:
                async with self._client.stream("POST", "/chat/completions", json=body) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_raw():
                        if chunk:
                            yielded_any = True
                            yield chunk
                return
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                delay = _retry_delay(exc, attempt)
                if yielded_any or attempt >= _RETRIES or delay is None:
                    raise
                await asyncio.sleep(delay)

    async def embeddings(self, payload: dict) -> dict:
        async def _once() -> dict:
            r = await self._client.post("/embeddings", json=payload)
            r.raise_for_status()
            return r.json()

        return await self._with_retry(_once)

    # ---- lifecycle (N/A for a remote API) --------------------------------
    async def pull(self, model: str) -> AsyncIterator[bytes]:
        yield (b'{"status":"error: a remote (openai) backend cannot pull models; '
               b'the provider hosts them"}\n')

    async def delete(self, model: str) -> None:
        raise NotImplementedError("delete is not supported for a remote (openai) backend")

    async def aclose(self) -> None:
        await self._client.aclose()
