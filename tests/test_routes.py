"""Route behaviour with a fake backend — no Ollama required.

Covers role resolution end-to-end, OpenAI response shapes, capability
gating, and embeddings defaulting. The live smoke (smoke_live.py) exercises
the real Ollama path.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from hearth import config as cfgmod
from hearth.app import create_app
from hearth.backends.base import Capabilities, ModelInfo


class FakeBackend:
    def __init__(self, name="ollama", caps: Capabilities | None = None):
        self.name = name
        self.base_url = "http://fake"
        self.capabilities = caps or Capabilities(
            chat=True, embeddings=True, streaming=True, tools=True, vision=True, json_mode=True
        )
        self.last_payload: dict | None = None

    async def health(self):
        return True

    async def list_models(self):
        return [ModelInfo(id="qwen2.5:14b-instruct-q4_K_M"), ModelInfo(id="nomic-embed-text")]

    async def chat(self, payload):
        self.last_payload = payload
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
        }

    async def chat_stream(self, payload):
        self.last_payload = payload
        for tok in ("he", "llo"):
            frame = {"choices": [{"delta": {"content": tok}}]}
            yield f"data: {json.dumps(frame)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    # Native (think-aware) path — stands in for Ollama's /api/chat.
    async def chat_native(self, payload, think):
        self.last_payload = payload
        self.last_think = think
        return {
            "id": "chatcmpl-x", "object": "chat.completion", "model": payload["model"],
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "native"},
                         "finish_reason": "stop"}],
        }

    async def chat_stream_native(self, payload, think):
        self.last_payload = payload
        self.last_think = think
        yield b'data: {"choices":[{"delta":{"content":"nat"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def embeddings(self, payload):
        self.last_payload = payload
        return {"object": "list", "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}], "model": payload["model"]}

    async def pull(self, model):
        yield b'{"status":"success"}\n'

    async def delete(self, model):
        return None

    async def aclose(self):
        return None


def _client(backend=None, **kw):
    cfg = cfgmod.default_config()
    backend = backend or FakeBackend()
    app = create_app(cfg, allow_any_host=True, backends_override={"ollama": backend}, **kw)
    return TestClient(app), backend


def test_models_lists_concrete_and_roles():
    client, _ = _client()
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "qwen2.5:14b-instruct-q4_K_M" in ids  # concrete from backend
    assert "primary_chat" in ids and "embedding" in ids  # role aliases


def test_chat_resolves_role_to_concrete_model():
    client, backend = _client()
    r = client.post("/v1/chat/completions", json={"model": "primary_chat", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    # The backend must have received the concrete model id, not the alias —
    # i.e. whatever primary_chat is bound to in the default config.
    expected = cfgmod.default_config().roles["primary_chat"].model
    assert backend.last_payload["model"] == expected
    assert r.json()["choices"][0]["message"]["content"] == "hi"


def test_think_field_routes_to_native_and_is_stripped():
    client, backend = _client()
    r = client.post("/v1/chat/completions",
                    json={"model": "primary_chat", "think": False,
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert backend.last_think is False                 # native path was taken
    assert r.json()["choices"][0]["message"]["content"] == "native"
    assert "think" not in backend.last_payload         # extension stripped from backend payload


def test_no_think_field_uses_openai_passthrough():
    client, backend = _client()
    backend.last_think = "untouched"
    r = client.post("/v1/chat/completions",
                    json={"model": "primary_chat",
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert backend.last_think == "untouched"           # native path NOT taken
    assert r.json()["choices"][0]["message"]["content"] == "hi"


def test_think_field_routes_to_native_streaming():
    client, backend = _client()
    with client.stream("POST", "/v1/chat/completions",
                       json={"model": "fast_chat", "stream": True, "think": False,
                             "messages": []}) as r:
        body = b"".join(r.iter_bytes())
    assert backend.last_think is False
    assert b'"content":"nat"' in body and b"[DONE]" in body


def test_chat_streaming_passthrough():
    client, _ = _client()
    with client.stream("POST", "/v1/chat/completions", json={"model": "fast_chat", "stream": True, "messages": []}) as r:
        body = b"".join(r.iter_bytes())
    assert b"data:" in body and b"[DONE]" in body


def test_embeddings_defaults_to_embedding_role():
    client, backend = _client()
    r = client.post("/v1/embeddings", json={"input": "hello"})  # no model
    assert r.status_code == 200
    assert backend.last_payload["model"] == "nomic-embed-text"
    assert len(r.json()["data"][0]["embedding"]) == 3


def test_tools_rejected_when_backend_lacks_capability():
    no_tools = FakeBackend(caps=Capabilities(chat=True, streaming=True, tools=False))
    client, _ = _client(backend=no_tools)
    r = client.post("/v1/chat/completions", json={"model": "primary_chat", "messages": [], "tools": [{"type": "function"}]})
    assert r.status_code == 400
    assert "tool" in r.json()["error"]["message"].lower()


def test_admin_version_and_server_header():
    client, _ = _client()
    r = client.get("/admin/version")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "hearth"
    assert body["contract"] == "0.1"
    assert body["engine"].startswith("hearth/")
    # Server header identifies hearth on every passed-through response.
    assert r.headers.get("server", "").startswith("hearth/")
    assert client.get("/v1/models").headers.get("server", "").startswith("hearth/")


def test_unbound_role_chat_is_clear_error():
    # 'vision' isn't bound in defaults -> falls to literal routing -> backend
    # gets "vision" as a model id, which is fine here (fake accepts anything).
    # But a role bound to a missing backend should 503.
    cfg = cfgmod.default_config()
    cfg.roles["primary_chat"] = cfgmod.RoleBinding(model="x", backend="ghost")
    app = create_app(cfg, allow_any_host=True, backends_override={"ollama": FakeBackend()})
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={"model": "primary_chat", "messages": []})
    assert r.status_code == 503
