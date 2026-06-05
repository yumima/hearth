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
import secrets
import time
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

    # ---- chat with explicit reasoning control (native /api/chat) ----------
    #
    # Ollama's OpenAI `/v1` surface can't toggle a reasoning model's thinking;
    # only the native `/api/chat` honours a `think` boolean. When a consumer
    # sets `think` we route here and translate the native NDJSON back into the
    # OpenAI shapes the rest of the stack already speaks (message.thinking ->
    # reasoning, message.content -> content) so callers see no difference.
    def _to_native(self, payload: dict, think: bool, stream: bool) -> dict:
        body = {
            "model": payload["model"],
            "messages": payload.get("messages", []),
            "stream": stream,
            "think": think,
        }
        if payload.get("tools"):
            body["tools"] = payload["tools"]
        opts: dict = {}
        if payload.get("temperature") is not None:
            opts["temperature"] = payload["temperature"]
        if payload.get("top_p") is not None:
            opts["top_p"] = payload["top_p"]
        if payload.get("max_tokens"):
            opts["num_predict"] = payload["max_tokens"]
        if opts:
            body["options"] = opts
        return body

    @staticmethod
    def _usage(obj: dict) -> dict:
        pt, ct = obj.get("prompt_eval_count") or 0, obj.get("eval_count") or 0
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

    @staticmethod
    def _err_text(err) -> str:
        if isinstance(err, dict):
            return err.get("message") or err.get("error") or str(err)
        return str(err)

    async def chat_native(self, payload: dict, think: bool) -> dict:
        r = await self._client.post("/api/chat", json=self._to_native(payload, think, False))
        r.raise_for_status()
        obj = r.json()
        # Ollama can answer 200 with an in-band error object — turn it into an
        # exception so the route surfaces a 502 instead of an empty completion.
        if obj.get("error"):
            raise RuntimeError(self._err_text(obj["error"]))
        msg = obj.get("message") or {}
        out_msg: dict = {"role": msg.get("role") or "assistant", "content": msg.get("content") or ""}
        if msg.get("thinking"):
            out_msg["reasoning"] = msg["thinking"]
        return {
            "id": "chatcmpl-" + secrets.token_hex(6),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": obj.get("model") or payload["model"],
            "choices": [{"index": 0, "message": out_msg,
                         "finish_reason": obj.get("done_reason") or "stop"}],
            "usage": self._usage(obj),
        }

    async def chat_stream_native(self, payload: dict, think: bool) -> AsyncIterator[bytes]:
        cid, created = "chatcmpl-" + secrets.token_hex(6), int(time.time())
        body = self._to_native(payload, think, True)
        done_emitted = False
        async with self._client.stream("POST", "/api/chat", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                # Ollama may emit an {"error": ...} line over an already-200
                # stream (model load failed, OOM, …) and close with no `done`
                # frame — surface it as an OpenAI error frame, don't swallow.
                if obj.get("error"):
                    yield f"data: {json.dumps({'error': {'message': self._err_text(obj['error'])}})}\n\n".encode()
                    break
                msg = obj.get("message") or {}
                delta: dict = {}
                if msg.get("thinking"):
                    delta["reasoning"] = msg["thinking"]
                if msg.get("content"):
                    delta["content"] = msg["content"]
                done = bool(obj.get("done"))
                if delta or done:
                    chunk = {
                        "id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": obj.get("model") or body["model"],
                        "choices": [{"index": 0, "delta": delta,
                                     "finish_reason": (obj.get("done_reason") or "stop") if done else None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                if done:
                    yield b"data: [DONE]\n\n"
                    done_emitted = True
                    break
        # Always terminate the SSE stream, even if Ollama closed it without a
        # clean `done` frame (error line above, truncation, dropped connection)
        # — OpenAI-SDK consumers wait on the [DONE] sentinel.
        if not done_emitted:
            yield b"data: [DONE]\n\n"

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
