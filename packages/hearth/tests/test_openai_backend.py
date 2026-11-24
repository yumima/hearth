"""Remote-backend behaviour: transient-error retry.

Hosted providers shed load with a 503/429 and expect the caller to retry.
Without that, the gateway turns a blip the provider considers routine into a
hard 502 at the client — measured against Gemini, where roughly one call in
three returned 503 and the immediate retry succeeded.
"""

import asyncio

import httpx
import pytest

from hearth.backends.openai import OpenAIBackend


def _backend(handler) -> OpenAIBackend:
    b = OpenAIBackend("x", "https://example.test/v1", "k")
    b._client = httpx.AsyncClient(
        base_url="https://example.test/v1", transport=httpx.MockTransport(handler)
    )
    return b


def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json={"ok": True})

    assert asyncio.run(_backend(handler).chat({"model": "m"})) == {"ok": True}
    assert calls["n"] == 3


def test_does_not_retry_auth_error():
    """A bad key is not transient — retrying only delays a clear error."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_backend(handler).chat({"model": "m"}))
    assert calls["n"] == 1


def test_retries_are_bounded():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_backend(handler).chat({"model": "m"}))
    assert calls["n"] == 3  # one attempt + _RETRIES


async def _abody():
    yield b'data: {"x":1}\n\n'


def test_stream_retries_before_first_byte():
    """Retry may only re-open the stream, never replay bytes a client saw."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503)
        # An ASYNC iterator keeps the response streamable: httpx marks a bytes
        # body as already-consumed (aiter_raw() then refuses it), and an async
        # client rejects a plain sync iterator.
        return httpx.Response(200, content=_abody())

    async def drain():
        return b"".join([c async for c in _backend(handler).chat_stream({"model": "m"})])

    assert b'data: {"x":1}' in asyncio.run(drain())
    assert calls["n"] == 2


def test_list_models_is_cached():
    """/v1/models fans out to every backend; a polling consumer must not turn
    each poll into an outbound provider call."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"data": [{"id": "gemini-3.7-flash"}]})

    b = _backend(handler)
    first = asyncio.run(b.list_models())
    second = asyncio.run(b.list_models())
    assert [m.id for m in first] == ["gemini-3.7-flash"]
    assert [m.id for m in second] == ["gemini-3.7-flash"]
    assert calls["n"] == 1, f"second call should hit cache, made {calls['n']} requests"


def test_list_models_serves_stale_on_error():
    """A blip should not empty a consumer's model dropdown."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"data": [{"id": "m1"}]})
        return httpx.Response(503)

    b = _backend(handler)
    assert [m.id for m in asyncio.run(b.list_models())] == ["m1"]
    b._models_cache = (0.0, b._models_cache[1])  # force expiry
    assert [m.id for m in asyncio.run(b.list_models())] == ["m1"]  # stale, not empty


def test_list_models_empty_without_cache():
    """No cache and a failing provider → empty, not a crash."""
    b = _backend(lambda request: httpx.Response(503))
    assert asyncio.run(b.list_models()) == []


def test_quota_429_is_not_blindly_retried():
    """Retrying a rate/quota limit on our own schedule spends more of the exact
    budget that just ran out. Google sends no Retry-After, so: fail fast."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_backend(handler).chat({"model": "m"}))
    assert calls["n"] == 1, f"429 burned {calls['n']} quota units instead of 1"


def test_429_with_short_retry_after_is_honoured():
    """When the provider does say how long to wait, respect it."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"ok": True})

    assert asyncio.run(_backend(handler).chat({"model": "m"})) == {"ok": True}
    assert calls["n"] == 2


def test_429_with_long_retry_after_gives_up():
    """A 10-minute wait is not something to block a chat request on."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, headers={"retry-after": "600"})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_backend(handler).chat({"model": "m"}))
    assert calls["n"] == 1


def test_stream_does_not_replay_after_first_byte():
    """A mid-stream TransportError must NOT be retried: re-issuing the POST
    restarts from byte 0 and the client sees the answer's opening twice."""
    calls = {"n": 0}

    async def _body_then_die():
        yield b'data: {"chunk":"hello"}\n\n'
        raise httpx.ReadError("connection dropped mid-stream")

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=_body_then_die())

    async def drain():
        out = b""
        async for c in _backend(handler).chat_stream({"model": "m"}):
            out += c
        return out

    with pytest.raises(httpx.ReadError):
        asyncio.run(drain())
    assert calls["n"] == 1, f"stream was re-issued {calls['n']}x — bytes would be replayed"
