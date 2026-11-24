"""Agent-loop tests — the backend is faked (so it's deterministic) but the MCP
host + tool execution are REAL (the self-contained stdio test server)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from mantel import app as appmod
from mantel import config as cfgmod
from mantel.mcp_host import ServerConfig

_SERVER = str(Path(__file__).parent / "_mcp_test_server.py")


def _sse(obj: dict) -> str:
    return json.dumps(obj)


async def _fake_with_tool(prov, req):
    """Round 1: stream a tool_call for t__add. Round 2 (tool result present):
    stream the final answer using the result."""
    if any(m.get("role") == "tool" for m in req["messages"]):
        yield _sse({"choices": [{"delta": {"content": "The sum is "}}]})
        yield _sse({"choices": [{"delta": {"content": "5."}, "finish_reason": "stop"}]})
        return
    yield _sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "t__add", "arguments": ""}}]}}]})
    yield _sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": '{"a": 2, "b": 3}'}}]}}]})
    yield _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})


async def _fake_plain(prov, req):
    yield _sse({"choices": [{"delta": {"content": "hello"}}]})
    yield _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})


def _post_chat(app) -> str:
    with TestClient(app, base_url="http://localhost") as client:  # `with` runs lifespan → starts the MCP host in this loop
        with client.stream("POST", "/api/chat",
                           json={"model": "x", "messages": [{"role": "user", "content": "add 2 and 3"}]}) as r:
            return b"".join(r.iter_bytes()).decode("utf-8")


def _answer(body: str) -> str:
    """Reconstruct the streamed assistant text from the SSE content frames."""
    out = ""
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except ValueError:
            continue
        delta = (obj.get("choices") or [{}])[0].get("delta", {})
        out += delta.get("content", "") or ""
    return out


def test_agent_loop_executes_a_tool(monkeypatch):
    monkeypatch.setattr(appmod, "_backend_chat_stream", _fake_with_tool)
    cfg = cfgmod.Config()
    cfg.mcp_servers = {"t": ServerConfig(command=sys.executable, args=[_SERVER],
                                         enabled=True, auto_approve=["add"])}
    body = _post_chat(appmod.create_app(cfg))
    assert '"mantel_tool"' in body                    # tool activity surfaced
    assert "t__add" in body                           # the right tool
    assert '"phase": "result"' in body and "5" in body  # executed → 5
    assert _answer(body) == "The sum is 5."           # streamed final answer (reassembled)


def test_plain_chat_when_no_mcp_servers(monkeypatch):
    monkeypatch.setattr(appmod, "_backend_chat_stream", _fake_plain)
    body = _post_chat(appmod.create_app(cfgmod.Config()))   # no mcp_servers
    assert _answer(body) == "hello" and '"mantel_tool"' not in body


def test_enable_disable_server_live(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))   # don't touch the real config
    cfg = cfgmod.Config()
    cfg.mcp_servers = {"t": ServerConfig(command=sys.executable, args=[_SERVER],
                                         enabled=False, auto_approve=["add", "echo"])}
    with TestClient(appmod.create_app(cfg), base_url="http://localhost") as client:
        assert client.get("/api/tools").json()["tools"] == []          # disabled → no tools
        servers = client.get("/api/servers").json()["servers"]
        assert servers and servers[0]["name"] == "t" and servers[0]["enabled"] is False

        r = client.post("/api/mcp/t/enable").json()                    # enable → live reload
        assert r["ok"] is True and r["enabled"] is True and r["tools"] >= 2
        assert "t__add" in {x["name"] for x in client.get("/api/tools").json()["tools"]}

        r2 = client.post("/api/mcp/t/disable").json()                  # disable → tools gone
        assert r2["enabled"] is False and r2["tools"] == 0
        assert client.get("/api/tools").json()["tools"] == []


def test_management_endpoint_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with TestClient(appmod.create_app(cfgmod.Config()), base_url="http://localhost") as client:
        assert client.post("/api/mcp/nope/enable").status_code == 404
        assert client.post("/api/provider/nope").status_code == 404
        assert client.post("/api/provider/hearth").json()["ok"] is True
