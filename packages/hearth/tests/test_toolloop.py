"""Tool-loop plumbing: SSE parsing and tool-call assembly.

These guard the failure modes that are invisible in a smoke test — a frame
split across chunk boundaries, or two tool calls merged into one — because both
produce a *plausible* completion rather than an error.

Async helpers are driven with asyncio.run() from sync tests so the suite needs
no pytest-asyncio configuration.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hearth import toolloop  # noqa: E402


async def _agen(chunks):
    for c in chunks:
        yield c


def _collect(chunks):
    async def run():
        return [e async for e in toolloop._iter_sse(_agen(chunks))]
    return asyncio.run(run())


def test_iter_sse_reassembles_frames_split_across_chunks():
    """A backend may split an SSE frame mid-JSON. Per-chunk parsing would drop
    it — and it's the long tool-call frames that get split."""
    frame = json.dumps({"choices": [{"delta": {"content": "hi"}}]})
    raw = f"data: {frame}\n\n".encode()
    events = _collect([raw[:12], raw[12:30], raw[30:]])
    assert len(events) == 1
    assert events[0]["choices"][0]["delta"]["content"] == "hi"


def test_iter_sse_handles_multiple_frames_in_one_chunk_and_done():
    a = json.dumps({"choices": [{"delta": {"content": "a"}}]})
    b = json.dumps({"choices": [{"delta": {"content": "b"}}]})
    events = _collect([f"data: {a}\n\ndata: {b}\n\ndata: [DONE]\n\n".encode()])
    assert [e["choices"][0]["delta"]["content"] for e in events[:2]] == ["a", "b"]
    assert events[2] is None  # [DONE] sentinel


def test_iter_sse_skips_malformed_json_without_aborting():
    good = json.dumps({"choices": [{"delta": {"content": "ok"}}]})
    events = _collect([b"data: {not json\n\n", f"data: {good}\n\n".encode()])
    assert len(events) == 1
    assert events[0]["choices"][0]["delta"]["content"] == "ok"


def test_accumulate_joins_streamed_argument_fragments():
    """OpenAI streams one call across many chunks: id/name first, then
    `arguments` in pieces."""
    acc: dict = {}
    toolloop._accumulate(acc, [{"index": 0, "id": "c1",
                                "function": {"name": "web_search", "arguments": '{"qu'}}])
    toolloop._accumulate(acc, [{"index": 0, "function": {"arguments": 'ery":"x"}'}}])
    calls = toolloop._finalize(acc)
    assert len(calls) == 1
    assert calls[0]["id"] == "c1"
    assert calls[0]["function"]["name"] == "web_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "x"}


def test_accumulate_keeps_distinct_indices_separate():
    """Two concurrent calls must not have their arguments concatenated."""
    acc: dict = {}
    toolloop._accumulate(acc, [
        {"index": 0, "id": "a", "function": {"name": "web_search", "arguments": '{"query":"1"}'}},
        {"index": 1, "id": "b", "function": {"name": "web_fetch", "arguments": '{"url":"u"}'}},
    ])
    calls = toolloop._finalize(acc)
    assert [c["function"]["name"] for c in calls] == ["web_search", "web_fetch"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"url": "u"}


def test_finalize_synthesizes_missing_id():
    acc: dict = {}
    toolloop._accumulate(acc, [{"index": 0, "function": {"name": "web_search", "arguments": "{}"}}])
    assert toolloop._finalize(acc)[0]["id"].startswith("call_")


def test_parse_args_tolerates_garbage():
    assert toolloop._parse_args("") == {}
    assert toolloop._parse_args("not json") == {}
    assert toolloop._parse_args('["a"]') == {}  # non-object JSON
    assert toolloop._parse_args('{"a":1}') == {"a": 1}


# ── the loop itself ───────────────────────────────────────────────────────────

def _sse(obj) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode()


def _frames(out) -> list[dict]:
    """Decode emitted SSE bytes back into chunk objects ([DONE] dropped)."""
    frames = []
    for line in b"".join(out).decode().split("\n\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            frames.append(json.loads(line[6:]))
    return frames


def _run_stream(rounds, our=("web_search",), max_rounds=4):
    """Drive stream_with_tools over a scripted list of per-round backend streams."""
    seen_calls = []

    def stream_fn(body):
        chunks = rounds[min(len(seen_calls), len(rounds) - 1)]
        return _agen(chunks)

    async def dispatch(name, args):
        seen_calls.append((name, args))
        return f"result for {args.get('query')}"

    async def run():
        out = []
        async for c in toolloop.stream_with_tools(
                stream_fn, {"model": "m", "messages": [{"role": "user", "content": "q"}]},
                our, dispatch, max_rounds, progress="off"):
            out.append(c)
        return out

    return asyncio.run(run()), seen_calls


def test_engine_tool_is_executed_and_not_leaked_to_client():
    tool_round = [
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1",
             "function": {"name": "web_search", "arguments": '{"query":"x"}'}}]}}]}),
        b"data: [DONE]\n\n",
    ]
    answer_round = [
        _sse({"choices": [{"delta": {"content": "final answer"}}]}),
        b"data: [DONE]\n\n",
    ]
    out, calls = _run_stream([tool_round, answer_round])
    body = b"".join(out).decode()
    assert calls == [("web_search", {"query": "x"})]
    assert "final answer" in body
    # The client must never see the engine's own call.
    assert "web_search" not in body
    assert body.endswith("data: [DONE]\n\n")


def test_client_tool_is_forwarded_unexecuted():
    rounds = [[
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c9",
             "function": {"name": "set_light", "arguments": '{"on":true}'}}]}}]}),
        b"data: [DONE]\n\n",
    ]]
    out, calls = _run_stream(rounds)
    body = b"".join(out).decode()
    assert calls == []  # engine executed nothing
    assert "set_light" in body
    # Parse rather than substring-match: the wire encoding is compact, and an
    # assertion on spacing would be testing json.dumps, not the loop.
    finishes = [c["choices"][0]["finish_reason"] for c in _frames(out)]
    assert "tool_calls" in finishes


def test_mixed_batch_is_forwarded_whole_rather_than_half_executed():
    """Half-answering a tool_calls turn would leave the client no valid reply."""
    rounds = [[
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "a",
             "function": {"name": "web_search", "arguments": '{"query":"x"}'}},
            {"index": 1, "id": "b",
             "function": {"name": "set_light", "arguments": "{}"}}]}}]}),
        b"data: [DONE]\n\n",
    ]]
    out, calls = _run_stream(rounds)
    body = b"".join(out).decode()
    assert calls == []
    assert "set_light" in body and "web_search" in body


def test_loop_terminates_at_max_rounds():
    """A model that keeps calling tools must not spin forever."""
    forever = [
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c",
             "function": {"name": "web_search", "arguments": '{"query":"x"}'}}]}}]}),
        b"data: [DONE]\n\n",
    ]
    out, calls = _run_stream([forever], max_rounds=3)
    assert len(calls) == 3
    assert b"".join(out).decode().endswith("data: [DONE]\n\n")


def test_nonstream_loop_strips_unanswerable_engine_calls_at_the_cap():
    """Rounds exhausted mid-tool-call: handing those calls back would name
    tools the client cannot implement."""
    async def chat_fn(body):
        return {"choices": [{"message": {"role": "assistant", "content": "",
                                         "tool_calls": [{"id": "c", "type": "function",
                                                         "function": {"name": "web_search",
                                                                      "arguments": '{"query":"x"}'}}]},
                             "finish_reason": "tool_calls"}]}

    async def dispatch(name, args):
        return "result"

    resp = asyncio.run(toolloop.chat_with_tools(
        chat_fn, {"model": "m", "messages": []}, ("web_search",), dispatch, 2))
    choice = resp["choices"][0]
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"
    assert "stopped after 2 tool rounds" in choice["message"]["content"]


# ── provider state passthrough ────────────────────────────────────────────────
#
# Gemini 3 attaches an encrypted `extra_content.google.thought_signature` to a
# function call and REJECTS the next turn if it doesn't come back verbatim —
# the failure that broke multi-turn tool use in LiteLLM and Codex. The loop
# rebuilds assistant messages from parsed parts, so anything it doesn't model
# explicitly is dropped by construction unless carried deliberately.


_SIG = {"google": {"thought_signature": "EucCCuQCARFNMg9huJ2cOUou"}}


def test_accumulate_preserves_provider_fields_on_tool_calls():
    acc: dict = {}
    toolloop._accumulate(acc, [{
        "index": 0, "id": "call_1", "type": "function",
        "function": {"name": "web_search", "arguments": '{"q":'},
        "extra_content": _SIG,
    }])
    toolloop._accumulate(acc, [{
        "index": 0, "function": {"arguments": '"nemotron"}'},
    }])
    calls = toolloop._finalize(acc)
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == '{"q":"nemotron"}'
    assert calls[0]["extra_content"] == _SIG, "thought signature must survive"


def test_finalize_without_provider_fields_is_unchanged():
    """The common case stays byte-identical — no stray keys for other backends."""
    acc: dict = {}
    toolloop._accumulate(acc, [{
        "index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}])
    assert toolloop._finalize(acc) == [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]


def test_accumulate_keeps_parallel_calls_separate_with_own_signatures():
    acc: dict = {}
    toolloop._accumulate(acc, [
        {"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"},
         "extra_content": {"google": {"thought_signature": "SIG_A"}}},
        {"index": 1, "id": "b", "function": {"name": "g", "arguments": "{}"},
         "extra_content": {"google": {"thought_signature": "SIG_B"}}},
    ])
    calls = toolloop._finalize(acc)
    assert [c["id"] for c in calls] == ["a", "b"]
    assert calls[0]["extra_content"]["google"]["thought_signature"] == "SIG_A"
    assert calls[1]["extra_content"]["google"]["thought_signature"] == "SIG_B"
