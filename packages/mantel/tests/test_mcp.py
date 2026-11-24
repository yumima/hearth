"""MCP host tests — against a real, self-contained stdio server (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mantel.mcp_host import MCPHost, ServerConfig

_SERVER = str(Path(__file__).parent / "_mcp_test_server.py")


def _server(auto_approve):
    return {"t": ServerConfig(command=sys.executable, args=[_SERVER],
                              enabled=True, auto_approve=auto_approve)}


@pytest.fixture
async def host():
    h = MCPHost(_server(auto_approve=["echo", "add"]))
    await h.start()
    yield h
    await h.stop()


async def test_aggregates_tools_namespaced(host):
    names = {t["function"]["name"] for t in host.openai_tools()}
    assert "t__echo" in names and "t__add" in names
    assert host.has_tools() and not host.errors


async def test_calls_a_tool(host):
    r = await host.call("t__add", {"a": 2, "b": 3})
    assert r["ok"] is True and "5" in r["content"]
    r2 = await host.call("t__echo", {"text": "hi"})
    assert r2["ok"] is True and "echo: hi" in r2["content"]


async def test_unknown_tool_is_handled(host):
    r = await host.call("t__nope", {})
    assert r["ok"] is False and "unknown tool" in r["content"]


async def test_deny_by_default():
    h = MCPHost(_server(auto_approve=[]))   # nothing approved
    await h.start()
    try:
        r = await h.call("t__echo", {"text": "hi"})
        assert r["ok"] is False and r.get("denied") is True
        assert "not auto-approved" in r["content"]
    finally:
        await h.stop()
