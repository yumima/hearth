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
    # These tests pin the routing without a role context, which forces the native path.
    for role in cfg.roles.values():
        role.context = 0
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
    assert body["contract"] == "0.2"
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


def _warm_body(monkeypatch, tmp_path, context):
    """What _warm_chat posts to Ollama for a primary_chat bound at `context`."""
    import asyncio

    from hearth.routes import v1

    monkeypatch.setenv("HEARTH_HOME", str(tmp_path))
    cfg = cfgmod.load()
    cfg.roles["primary_chat"].context = context
    sent = {}

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json):
            sent.update(json)

    monkeypatch.setattr(v1.httpx, "AsyncClient", FakeClient)
    asyncio.run(v1._warm_chat(cfg))
    return sent


def test_warm_chat_loads_at_the_role_context(monkeypatch, tmp_path):
    # Warming at another context makes the next chat reload the model anyway.
    assert _warm_body(monkeypatch, tmp_path, 65536)["options"] == {"num_ctx": 65536}


def test_warm_chat_leaves_context_to_the_daemon_when_unset(monkeypatch, tmp_path):
    assert "options" not in _warm_body(monkeypatch, tmp_path, 0)


def test_nonstreaming_chat_stops_when_the_client_hangs_up():
    """Otherwise the abandoned request keeps the model busy and every later
    prompt queues behind it, which looks like the model not responding."""
    import asyncio

    class Hangs(FakeBackend):
        cancelled = False

        async def chat(self, payload):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                type(self).cancelled = True
                raise

    cfg = cfgmod.default_config()
    for role in cfg.roles.values():
        role.context = 0
    cfg.search.enabled = False
    app = create_app(cfg, allow_any_host=True, backends_override={"ollama": Hangs()})
    body = json.dumps({"model": "primary_chat",
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions", "query_string": b"",
             "root_path": "", "scheme": "http", "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 11435),
             "headers": [(b"host", b"127.0.0.1:11435"), (b"content-type", b"application/json")]}
    sent = []

    async def run():
        first = True
        hang_up_at = asyncio.get_running_loop().time() + 0.05

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            # is_disconnected() polls with an already-cancelled scope, so the
            # hang-up must be ready without awaiting once it has happened.
            if asyncio.get_running_loop().time() < hang_up_at:
                await asyncio.sleep(3600)
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await asyncio.wait_for(app(scope, receive, send), timeout=5)

    asyncio.run(run())
    assert Hangs.cancelled
    assert sent[0]["status"] == 499



# ── web search: only when the request asks ────────────────────────────────────

def _tool_names(backend):
    return [t["function"]["name"] for t in backend.last_payload.get("tools") or []]


def _chat(cfg=None, **body):
    b = FakeBackend()
    app = create_app(cfg or cfgmod.default_config(), backends_override={"ollama": b})
    r = TestClient(app, base_url="http://localhost").post("/v1/chat/completions", json={
        "model": "primary_chat", "messages": [{"role": "user", "content": "hi"}], **body})
    assert r.status_code == 200
    return b


def test_web_tools_are_not_attached_unasked():
    tool = {"type": "function", "function": {"name": "get_holdings", "parameters": {}}}
    for body in ({}, {"tools": [tool]}, {"web_search": False}, {"web_search": None},
                 {"web_search": "false"}, {"web_search": 1}, {"stream": True, "tools": [tool]}):
        b = _chat(**body)
        assert not {"web_search", "web_fetch"} & set(_tool_names(b)), body
        assert "web_search" not in b.last_payload  # a hearth field, not the backend's
        assert len(b.last_payload["messages"]) == 1  # nor its preamble


def test_web_tools_join_the_clients_own_when_asked():
    tool = {"type": "function", "function": {"name": "mine", "parameters": {}}}
    assert _tool_names(_chat(web_search=True, tools=[tool])) == ["mine", "web_search", "web_fetch"]


def test_asking_does_not_override_search_turned_off():
    cfg = cfgmod.default_config()
    cfg.search.enabled = False
    assert _tool_names(_chat(cfg, web_search=True)) == []


def test_a_config_from_when_search_was_automatic_attaches_nothing(monkeypatch, tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("search:\n  enabled: true\n  auto: true\n")
    monkeypatch.setenv("HEARTH_CONFIG", str(p))
    cfg = cfgmod.load()
    assert _tool_names(_chat(cfg)) == []
    assert "auto" not in cfg.to_yaml_dict()["search"]


def test_a_think_off_stream_gets_the_tools_it_asks_for():
    """What `hearth chat` and the window send in their fast lane."""
    assert _tool_names(_chat(web_search=True, stream=True, think=False)) == ["web_search", "web_fetch"]


def _loaded(monkeypatch, tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    monkeypatch.setenv("HEARTH_CONFIG", str(p))
    return cfgmod.load()


def test_turning_automatic_search_off_still_keeps_our_chats_off_the_web(monkeypatch, tmp_path):
    assert _loaded(monkeypatch, tmp_path, "search: {auto: false}\n").search.chat is False
    assert _loaded(monkeypatch, tmp_path, "search: {auto: true}\n").search.chat is True
    assert _loaded(monkeypatch, tmp_path, "{}\n").search.chat is True
    cfg = _loaded(monkeypatch, tmp_path, "search: {auto: false, chat: true}\n")
    assert cfg.search.chat is True  # the newer key wins
    cfg.search.chat = False
    cfgmod.save(cfg)
    assert cfgmod.load().search.chat is False


def test_the_terminal_chat_asks_as_configured(monkeypatch, capsys):
    import io

    from hearth import cli

    for chat in (True, False):
        cfg = cfgmod.default_config()
        cfg.search.chat = chat
        sent = []

        def stream(method, url, json=None, **kw):
            sent.append(json)
            raise cli.httpx.ConnectError("stop here")

        monkeypatch.setattr(cli.cfgmod, "load", lambda *a, **k: cfg)
        monkeypatch.setattr(cli.httpx, "stream", stream)
        monkeypatch.setattr(cli.httpx, "get", lambda *a, **k: None)
        monkeypatch.setattr(cli, "_chat_models", lambda base: (["primary_chat"], []))
        monkeypatch.setattr("builtins.input", iter_input("hello"))
        monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))
        cli.main(["chat"])
        assert [s["web_search"] for s in sent] == [chat]


def iter_input(*lines):
    it = iter(lines)

    def read(*_):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read
