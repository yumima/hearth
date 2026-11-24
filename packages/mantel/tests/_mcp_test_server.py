"""A tiny self-contained MCP server (stdio) for tests — no network, no npx.

Exposes `echo` and `add` so the host/agent-loop tests can connect to a real MCP
server over stdio and exercise list_tools / call_tool deterministically.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mantel-test")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the input text back."""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
