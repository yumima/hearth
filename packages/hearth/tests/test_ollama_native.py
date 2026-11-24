"""Native (/api/chat) translation in the Ollama backend.

The think-toggle path talks to Ollama's native API and translates its NDJSON
back into OpenAI shapes. These tests drive that translation with a fake httpx
client — no Ollama needed — covering the happy path and the in-band error
case (HTTP 200 + an {"error": ...} line with no `done` frame) that must still
surface and terminate the stream.
"""

from __future__ import annotations

import asyncio

from hearth.backends.ollama import OllamaBackend


class _FakeStream:
    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln


class _FakeResp:
    def __init__(self, obj: dict):
        self._obj = obj

    def raise_for_status(self):
        pass

    def json(self):
        return self._obj


class _FakeClient:
    def __init__(self, lines=None, post_obj=None):
        self._lines = lines or []
        self._post_obj = post_obj or {}
        self.posted: dict | None = None

    def stream(self, method, url, json=None):
        self.posted = json
        return _FakeStream(self._lines)

    async def post(self, url, json=None):
        self.posted = json
        return _FakeResp(self._post_obj)


def _backend(client) -> OllamaBackend:
    b = OllamaBackend("ollama", "http://127.0.0.1:0")
    asyncio.run(b._client.aclose())  # drop the real client created in __init__
    b._client = client
    return b


def _drain(agen) -> bytes:
    async def collect():
        return b"".join([c async for c in agen])
    return asyncio.run(collect())


def test_native_stream_translates_thinking_and_content():
    client = _FakeClient(lines=[
        '{"message":{"role":"assistant","thinking":"hmm"}}',
        '{"message":{"content":"4"}}',
        '{"message":{"content":""},"done":true,"done_reason":"stop","eval_count":3}',
    ])
    out = _drain(_backend(client).chat_stream_native({"model": "m", "messages": []}, False))
    assert b'"reasoning": "hmm"' in out      # thinking -> reasoning
    assert b'"content": "4"' in out          # content -> content
    assert b'"finish_reason": "stop"' in out
    assert out.count(b"data: [DONE]") == 1   # terminated exactly once
    assert client.posted["think"] is False   # think forwarded to native body


def test_native_stream_surfaces_error_line_and_still_terminates():
    # HTTP 200 then an error line, no `done` frame — the pre-fix blind spot.
    client = _FakeClient(lines=['{"error":"model failed to load"}'])
    out = _drain(_backend(client).chat_stream_native({"model": "m", "messages": []}, True))
    assert b"model failed to load" in out
    assert b'"error"' in out
    assert b"data: [DONE]" in out            # stream still terminated


def test_native_stream_terminates_without_done_frame():
    # Stream ends cleanly but Ollama never sent done:true (truncation / drop).
    client = _FakeClient(lines=['{"message":{"content":"hi"}}'])
    out = _drain(_backend(client).chat_stream_native({"model": "m", "messages": []}, False))
    assert b'"content": "hi"' in out
    assert out.count(b"data: [DONE]") == 1


def test_native_nonstream_raises_on_error_object():
    client = _FakeClient(post_obj={"error": "boom"})
    backend = _backend(client)
    try:
        asyncio.run(backend.chat_native({"model": "m", "messages": []}, False))
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("expected RuntimeError on in-band error object")


def test_native_nonstream_translates_message():
    client = _FakeClient(post_obj={
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": "4", "thinking": "t"},
        "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2,
    })
    out = asyncio.run(_backend(client).chat_native({"model": "m", "messages": []}, False))
    assert out["choices"][0]["message"]["content"] == "4"
    assert out["choices"][0]["message"]["reasoning"] == "t"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 7
    assert out["object"] == "chat.completion"
