"""Mantel server tests — UI route + the /api proxy, against an UNREACHABLE
backend so they're deterministic and fast (no real engine needed)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mantel import config as cfgmod
from mantel.app import create_app


def _client() -> TestClient:
    cfg = cfgmod.Config()
    # Point the backend at a closed port so proxy calls fail fast (refused),
    # exercising mantel's graceful-degradation paths without a running engine.
    cfg.provider = "hearth"
    cfg.providers = {"hearth": cfgmod.Provider(base_url="http://127.0.0.1:1/v1")}
    # base_url=localhost so the localhost-only guard accepts the request.
    return TestClient(create_app(cfg), base_url="http://localhost")


def test_local_guard_blocks_rebinding_and_csrf():
    app = create_app(cfgmod.Config())
    assert TestClient(app, base_url="http://evil.example.com").get("/api/health").status_code == 403
    ok = TestClient(app, base_url="http://localhost")
    assert ok.get("/api/health").status_code == 200
    # cross-origin (CSRF) POST is refused even with a localhost Host
    assert ok.post("/api/provider/hearth", headers={"origin": "http://evil.example.com"}).status_code == 403


def test_index_serves_ui():
    r = _client().get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>Mantel</title>" in r.text
    assert "/api/chat" in r.text                 # same-origin wiring
    assert 'src="http' not in r.text and "<link" not in r.text  # self-contained


def test_health():
    r = _client().get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_config_exposes_active_backend():
    body = _client().get("/api/config").json()
    assert body["provider"] == "hearth"
    assert body["model"]
    assert body["backend"].startswith("http://")
    assert "hearth" in body["providers"]   # drives the /provider palette completion


def test_models_degrades_gracefully_when_backend_down():
    r = _client().get("/api/models")
    assert r.status_code == 502                  # backend unreachable…
    assert r.json().get("data") == []            # …but a usable shape, no crash


def test_chat_streams_an_error_frame_when_backend_down():
    with _client().stream("POST", "/api/chat",
                          json={"model": "primary_chat", "messages": [{"role": "user", "content": "hi"}]}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_bytes())
    assert b"error" in body and b"unreachable" in body  # surfaced, not swallowed


def test_chat_persistence_crud(tmp_path, monkeypatch):
    """Conversations save, list, reload (multimodal content intact), and delete —
    against an isolated data dir so the real chats.db is never touched."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    c = _client()

    msgs = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": [
            {"type": "text", "text": "what's this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
    ]
    assert c.put("/api/chats/conv-1", json={"title": "hello there", "messages": msgs}).status_code == 200

    listed = c.get("/api/chats").json()["chats"]
    assert len(listed) == 1 and listed[0]["id"] == "conv-1" and listed[0]["messages"] == 3

    got = c.get("/api/chats/conv-1").json()
    assert got["title"] == "hello there" and len(got["messages"]) == 3
    # multimodal content round-trips as a list, not a stringified blob
    assert isinstance(got["messages"][2]["content"], list)
    assert got["messages"][2]["content"][1]["type"] == "image_url"

    # PUT replaces (upsert), doesn't append
    c.put("/api/chats/conv-1", json={"title": "renamed", "messages": msgs[:1]})
    got2 = c.get("/api/chats/conv-1").json()
    assert got2["title"] == "renamed" and len(got2["messages"]) == 1

    assert c.delete("/api/chats/conv-1").json()["ok"] is True
    assert c.get("/api/chats").json()["chats"] == []
    assert c.get("/api/chats/conv-1").status_code == 404


def test_chat_persistence_tolerates_malformed_messages(tmp_path, monkeypatch):
    """A malformed messages payload must not 500 (non-dict items are dropped)."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    c = _client()
    r = c.put("/api/chats/bad", json={"title": "x", "messages": ["not-a-dict", {"role": "user", "content": "ok"}]})
    assert r.status_code == 200
    assert len(c.get("/api/chats/bad").json()["messages"]) == 1


def test_hearth_engine_tools_surface_as_tool_events(monkeypatch):
    """hearth runs web_search/web_fetch inside the completion and announces each
    on a non-standard `hearth_tool` delta field. The proxy rebuilds frames field
    by field, so anything it doesn't know about is dropped — which left the
    engine's work invisible in the UI and made answers look unsourced."""
    import json

    from mantel import app as appmod

    frames = [
        {"choices": [{"delta": {"hearth_tool": {
            "phase": "call", "id": "call_1", "name": "web_search",
            "arguments": {"query": "debian release"}}, "reasoning": "🔎 Searching…\n"}}]},
        {"choices": [{"delta": {"hearth_tool": {
            "phase": "result", "id": "call_1", "name": "web_search",
            "ok": True, "content": "5 results"}}}]},
        {"choices": [{"delta": {"content": "Debian 13."}}]},
    ]

    async def fake_stream(prov, req, client=None):
        for f in frames:
            yield json.dumps(f)

    monkeypatch.setattr(appmod, "_backend_chat_stream", fake_stream)
    with _client().stream("POST", "/api/chat",
                          json={"model": "primary_chat",
                                "messages": [{"role": "user", "content": "hi"}]}) as r:
        body = b"".join(r.iter_bytes()).decode()

    events = [json.loads(ln[5:]) for ln in body.splitlines()
              if ln.startswith("data:") and ln[5:].strip() not in ("", "[DONE]")]
    tools = [e["mantel_tool"] for e in events if "mantel_tool" in e]
    assert [t["phase"] for t in tools] == ["call", "result"]
    # Both phases must carry the SAME id or the UI cannot resolve "running…".
    assert tools[0]["id"] == tools[1]["id"] == "hearth:call_1"
    assert tools[0]["name"] == "web_search"
    assert tools[0]["arguments"]["query"] == "debian release"
    assert tools[1]["content"] == "5 results"
    # The answer itself still streams through untouched.
    assert "Debian 13." in body


# ── web search ────────────────────────────────────────────────────────────────

def _sent(monkeypatch, cfg, **body) -> dict:
    """The request /api/chat sends to the backend."""
    import json

    from mantel import app as appmod

    sent = {}

    async def fake_stream(prov, req, client=None):
        sent.update(req)
        yield json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(appmod, "_backend_chat_stream", fake_stream)
    r = TestClient(create_app(cfg), base_url="http://localhost").post("/api/chat", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}], **body})
    assert r.status_code == 200
    return sent


def test_hearth_is_asked_for_its_web_tools(monkeypatch):
    assert _sent(monkeypatch, cfgmod.Config())["web_search"] is True


def test_web_search_off_in_config_wins_over_the_request(monkeypatch):
    cfg = cfgmod.Config(web_search=False)
    assert _sent(monkeypatch, cfg)["web_search"] is False
    assert _sent(monkeypatch, cfg, web_search=True)["web_search"] is False
    assert _sent(monkeypatch, cfgmod.Config(), web_search=False)["web_search"] is False


def test_other_backends_never_get_the_hearth_field(monkeypatch):
    cfg = cfgmod.Config(provider="openai", providers={
        "openai": cfgmod.Provider(base_url="http://127.0.0.1:1/v1", api_key="k")})
    assert "web_search" not in _sent(monkeypatch, cfg)
    assert "web_search" not in _sent(monkeypatch, cfg, web_search=True)


def test_web_search_setting_survives_a_save_and_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert cfgmod.load().web_search is True            # no file
    cfgmod.save(cfgmod.Config(web_search=False))
    assert cfgmod.load().web_search is False
    cfgmod.config_path().write_text("provider: hearth\n")
    assert cfgmod.load().web_search is True            # a config from before the setting
    for bad in ("'no'", "nope", "0", "1", "null"):
        cfgmod.config_path().write_text(f"web_search: {bad}\n")
        assert cfgmod.load().web_search is False, bad


# ── voice choice ──────────────────────────────────────────────────────────────

def test_voice_list_degrades_when_backend_down():
    assert _client().get("/api/voices").json() == {"voices": [], "default": None, "selected": ""}


def test_chosen_voice_is_saved_and_spoken(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    c = _client()
    assert c.put("/api/voice", json={"voice": "kokoro:bm_george "}).json()["voice"] == "kokoro:bm_george"
    assert c.get("/api/voices").json()["selected"] == "kokoro:bm_george"
    assert c.get("/api/config").json()["tts_voice"] == "kokoro:bm_george"
    assert cfgmod.load().tts_voice == "kokoro:bm_george"


def test_voice_list_that_is_not_an_object_is_no_list(monkeypatch):
    """A backend whose /audio/voices answers something else must not break Settings."""
    import httpx

    async def get(self, url, **kw):
        return httpx.Response(200, json=["alloy"], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    assert _client().get("/api/voices").json()["voices"] == []
