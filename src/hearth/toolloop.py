"""Server-side tool loop for /v1/chat/completions.

Hosted providers run their built-in tools (web search, code execution) *inside*
the completion: the client sends one request and gets one answer, with the
search round-trips happening invisibly in between. Hearth does the same for the
tools defined in webtools.py, which is what lets an unmodified OpenAI SDK
client — or finterm, or mantel — get live data with no code of its own.

The two rules that keep this safe to bolt onto a passthrough gateway:

* **Only ours.** A tool call the engine does not own is forwarded to the client
  untouched, exactly as before. If a single turn mixes engine and client tools
  we forward the whole batch unexecuted, because we cannot half-answer a
  tool_calls turn and still leave the client a valid protocol state.
* **Bounded.** ``max_tool_rounds`` caps the loop; hitting it ends the turn with
  a note rather than looping until the context blows up.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from contextlib import aclosing
from typing import AsyncIterator, Awaitable, Callable

_DONE = b"data: [DONE]\n\n"
# Match the compact encoding every OpenAI-compatible server emits. The loop has
# to re-serialize frames (it rewrites id/model so a multi-round turn reads as
# one completion), and json.dumps' default ", "/": " spacing would make our
# output differ byte-for-byte from a plain passthrough.
_SEP = (",", ":")


# ── SSE plumbing ──────────────────────────────────────────────────────────────

async def _iter_sse(raw: AsyncIterator[bytes]):
    """Parse an SSE byte stream into (event_dict | None) — None is [DONE].

    Buffers across chunk boundaries: a backend is free to split a frame
    mid-JSON, and naive per-chunk parsing drops exactly the long tool-call
    frames we care about.
    """
    buf = b""
    async for chunk in raw:
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == b"[DONE]":
                    yield None
                    continue
                try:
                    yield json.loads(data)
                except (json.JSONDecodeError, ValueError):
                    continue


def _frame(cid: str, created: int, model: str, delta: dict,
           finish: str | None = None, usage: dict | None = None) -> bytes:
    obj = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        obj["usage"] = usage
    return ("data: " + json.dumps(obj, separators=_SEP) + "\n\n").encode()


# ── tool-call accumulation ────────────────────────────────────────────────────

def _accumulate(acc: dict, deltas: list) -> None:
    """Merge streamed tool_call deltas into ``acc`` keyed by index.

    OpenAI streams a tool call across many chunks: the first carries id/name,
    the rest append `arguments` fragments.
    """
    for d in deltas or []:
        idx = d.get("index", 0)
        slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        if d.get("id"):
            slot["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["arguments"] += fn["arguments"]


def _finalize(acc: dict) -> list[dict]:
    out = []
    for idx in sorted(acc):
        s = acc[idx]
        out.append({"id": s["id"] or f"call_{secrets.token_hex(6)}",
                    "type": "function",
                    "function": {"name": s["name"], "arguments": s["arguments"] or "{}"}})
    return out


def _parse_args(raw: str) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _label(name: str, args: dict) -> str:
    if name == "web_search":
        return f'Searching the web for "{args.get("query", "")}"'
    if name == "web_fetch":
        return f'Reading {args.get("url", "")}'
    return f"Running {name}"


def _summarize(name: str, result: str) -> str:
    """One short line describing what a tool returned, for the UI card."""
    if result.startswith("web_search failed") or result.startswith("web_fetch error"):
        return result.splitlines()[0][:160]
    if name == "web_search":
        # Results are numbered "1. title"; counting them beats echoing the blob.
        n = sum(1 for ln in result.splitlines() if re.match(r"^\d+\. ", ln))
        return f"{n} result{'s' if n != 1 else ''}"
    if name == "web_fetch":
        first = result.splitlines()[0] if result else ""
        return f"{first[:120]} · {len(result)} chars"
    return f"{len(result)} chars"


def _progress(cid: str, created: int, model: str, mode: str, phase: str,
              call_id: str, name: str, args: dict, summary: str = "") -> bytes | None:
    """A frame announcing an engine tool call starting or finishing.

    The structured ``hearth_tool`` object is what a UI renders as a tool card
    (mantel matches call→result on ``id``); the text line in content/reasoning
    is the fallback for clients that only know the OpenAI fields. Emitting a
    result phase as well as a call is what lets a UI show "running…" resolve to
    a real outcome instead of leaving a spinner that never completes.
    """
    if mode == "off":
        return None
    delta: dict = {"hearth_tool": {"phase": phase, "id": call_id, "name": name,
                                   **({"arguments": args} if phase == "call"
                                      else {"ok": True, "content": summary})}}
    if phase == "call":
        delta["content" if mode == "content" else "reasoning"] = f"🔎 {_label(name, args)}…\n"
    return _frame(cid, created, model, delta)


# ── streaming ─────────────────────────────────────────────────────────────────

async def stream_with_tools(
    stream_fn: Callable[[dict], AsyncIterator[bytes]],
    payload: dict,
    our_names: tuple[str, ...],
    dispatch: Callable[[str, dict], Awaitable[str]],
    max_rounds: int,
    progress: str = "reasoning",
) -> AsyncIterator[bytes]:
    messages = list(payload.get("messages") or [])
    cid, created = "chatcmpl-" + secrets.token_hex(6), int(time.time())
    model = payload.get("model", "")
    # Token usage is summed across rounds, not taken from the last one: the
    # search round-trips are real tokens the caller paid for, and reporting
    # only the final turn would understate a tool-heavy answer several-fold.
    usage: dict = {}

    for _round in range(max(1, max_rounds)):
        acc: dict = {}
        content_seen = ""
        finish: str | None = None
        body = {**payload, "messages": messages}
        # aclosing: we can leave this loop early (an error frame, or a client
        # tool batch), and an abandoned async generator would only release the
        # backend's HTTP stream whenever GC got to it.
        async with aclosing(_iter_sse(stream_fn(body))) as events:
            async for evt in events:
                if evt is None:
                    break
                if evt.get("error"):
                    yield ("data: " + json.dumps(evt, separators=_SEP) + "\n\n").encode()
                    yield _DONE
                    return
                if evt.get("usage"):
                    for k, v in evt["usage"].items():
                        if isinstance(v, int):
                            usage[k] = usage.get(k, 0) + v
                model = evt.get("model") or model
                choice = (evt.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if delta.get("tool_calls"):
                    # Hold these back: we don't yet know whether they're ours to
                    # run or the client's to receive.
                    _accumulate(acc, delta["tool_calls"])
                    continue
                if delta.get("content"):
                    content_seen += delta["content"]
                # Rewrite the id/model so a multi-round turn looks like one
                # completion to the client instead of several stitched together.
                if delta:
                    yield _frame(cid, created, model, delta)

        calls = _finalize(acc)
        if not calls:
            yield _frame(cid, created, model, {}, finish or "stop", usage)
            yield _DONE
            return

        if not all(c["function"]["name"] in our_names for c in calls):
            # Client-owned tools (possibly mixed with ours): hand the whole
            # batch over unexecuted and let the client drive the next turn.
            yield _frame(cid, created, model,
                         {"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]})
            yield _frame(cid, created, model, {}, "tool_calls", usage)
            yield _DONE
            return

        messages.append({"role": "assistant", "content": content_seen or None,
                         "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            args = _parse_args(c["function"]["arguments"])
            frame = _progress(cid, created, model, progress, "call", c["id"], name, args)
            if frame:
                yield frame
            result = await dispatch(name, args)
            frame = _progress(cid, created, model, progress, "result", c["id"], name, args,
                              _summarize(name, result))
            if frame:
                yield frame
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "name": name, "content": result})

    yield _frame(cid, created, model,
                 {"content": f"\n\n_(stopped after {max_rounds} tool rounds)_"},
                 "stop", usage)
    yield _DONE


# ── non-streaming ─────────────────────────────────────────────────────────────

async def chat_with_tools(
    chat_fn: Callable[[dict], Awaitable[dict]],
    payload: dict,
    our_names: tuple[str, ...],
    dispatch: Callable[[str, dict], Awaitable[str]],
    max_rounds: int,
) -> dict:
    messages = list(payload.get("messages") or [])
    resp: dict = {}
    for _round in range(max(1, max_rounds)):
        resp = await chat_fn({**payload, "messages": messages})
        msg = ((resp.get("choices") or [{}])[0].get("message")) or {}
        calls = msg.get("tool_calls") or []
        if not calls or not all(
                ((c.get("function") or {}).get("name") in our_names) for c in calls):
            return resp
        messages.append({"role": "assistant", "content": msg.get("content") or None,
                         "tool_calls": calls})
        for c in calls:
            fn = c.get("function") or {}
            name = fn.get("name", "")
            result = await dispatch(name, _parse_args(fn.get("arguments") or "{}"))
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                             "name": name, "content": result})

    # Rounds exhausted while the model was still calling engine tools. Handing
    # those calls to the client would be a dead end — they name tools it has no
    # implementation for and cannot return results to. Strip them so the caller
    # gets a terminated completion rather than an unanswerable one.
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    if msg.get("tool_calls"):
        msg = {**msg, "content": (msg.get("content") or "")
               + f"\n\n_(stopped after {max_rounds} tool rounds)_"}
        msg.pop("tool_calls", None)
        resp = {**resp, "choices": [{**choice, "message": msg, "finish_reason": "stop"}]}
    return resp
