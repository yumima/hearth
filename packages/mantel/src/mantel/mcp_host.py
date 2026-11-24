"""MCP host — connect to configured MCP servers, aggregate their tools, and
execute tool calls on the model's behalf.

mantel is an MCP *host*: for each enabled server it opens a long-lived stdio
session, lists the tools, and exposes them to the backend model as OpenAI
function tools (namespaced ``<server>__<tool>``). Tool calls route back to the
owning server.

Execution is **deny-by-default**: a tool only runs if its name matches one of
its server's ``auto_approve`` globs — anything else is refused with a message
the model (and user) sees. That keeps an autonomous model from touching files /
shells / networks you didn't explicitly allow. (An interactive approval surface
is a later milestone; this is the safe floor.)
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_CONNECT_TIMEOUT = 20.0  # per-server cap on connect+initialize+list_tools


@dataclass
class ServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = False
    auto_approve: list[str] = field(default_factory=list)   # globs over tool names


@dataclass
class _Tool:
    server: str
    name: str               # the tool's name on its server
    description: str
    schema: dict
    qualified: str          # OpenAI tool name exposed to the model: <server>__<tool>
    auto_approve: list[str]


def _sanitize(s: str) -> str:
    # OpenAI tool names must match [a-zA-Z0-9_-]; map anything else to "_".
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


class MCPHost:
    """Owns the live MCP sessions. Start it inside the app's lifespan (one task),
    call its tools from request handlers, stop it on shutdown."""

    def __init__(self, servers: dict[str, ServerConfig] | None = None):
        self._servers = servers or {}
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, _Tool] = {}     # qualified name -> _Tool
        self.errors: dict[str, str] = {}       # server name -> connection error

    async def start(self) -> None:
        for name, sc in self._servers.items():
            if not sc.enabled:
                continue
            try:
                params = StdioServerParameters(
                    command=sc.command, args=sc.args, env={**os.environ, **sc.env})
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                # Bound the handshake: a server that connects but never responds
                # to initialize must not hang start() (and thus app startup).
                await asyncio.wait_for(session.initialize(), timeout=_CONNECT_TIMEOUT)
                listed = await asyncio.wait_for(session.list_tools(), timeout=_CONNECT_TIMEOUT)
                self._sessions[name] = session
                for t in listed.tools:
                    q = f"{_sanitize(name)}__{_sanitize(t.name)}"[:64]
                    if q in self._tools:  # truncation/sanitize collision → disambiguate
                        q = f"{q[:60]}_{len(self._tools)}"[:64]
                    self._tools[q] = _Tool(
                        server=name, name=t.name, description=t.description or "",
                        schema=t.inputSchema or {"type": "object"},
                        qualified=q, auto_approve=list(sc.auto_approve))
            except Exception as e:  # a bad server must not sink the rest
                self.errors[name] = str(e)

    async def stop(self) -> None:
        try:
            await self._stack.aclose()
        except Exception:
            pass
        self._sessions.clear()
        self._tools.clear()

    def has_tools(self) -> bool:
        return bool(self._tools)

    def openai_tools(self) -> list[dict]:
        """The aggregated tools as OpenAI function-tool schemas."""
        return [{
            "type": "function",
            "function": {
                "name": t.qualified,
                "description": (f"[{t.server}] {t.description}").strip()[:1024],
                "parameters": t.schema,
            },
        } for t in self._tools.values()]

    def _approved(self, t: _Tool) -> bool:
        return any(fnmatch.fnmatch(t.name, pat) for pat in t.auto_approve)

    async def call(self, qualified: str, arguments: dict[str, Any] | None) -> dict:
        """Execute one tool call. Returns a dict with ``ok``, ``content``, and
        (on a policy refusal) ``denied`` — never raises."""
        t = self._tools.get(qualified)
        if t is None:
            return {"ok": False, "tool": qualified, "content": f"unknown tool '{qualified}'"}
        if not self._approved(t):
            return {"ok": False, "denied": True, "server": t.server, "tool": t.name,
                    "content": (f"Tool '{t.name}' on server '{t.server}' is not auto-approved. "
                                f"Add it to that server's auto_approve list to allow it.")}
        session = self._sessions.get(t.server)
        if session is None:
            return {"ok": False, "server": t.server, "tool": t.name,
                    "content": f"server '{t.server}' is not connected"}
        try:
            result = await session.call_tool(t.name, arguments or {})
            return {"ok": not result.isError, "server": t.server, "tool": t.name,
                    "content": _result_text(result)}
        except Exception as e:
            return {"ok": False, "server": t.server, "tool": t.name, "content": f"tool error: {e}"}


def _result_text(result) -> str:
    """Flatten an MCP CallToolResult's content blocks to text."""
    parts: list[str] = []
    for c in (result.content or []):
        if getattr(c, "type", None) == "text":
            parts.append(c.text)
        else:
            parts.append(str(getattr(c, "text", c)))
    return "\n".join(parts)
