"""mantel — a universal, MCP-native desktop chat client.

Talks to any OpenAI-compatible backend (and, later, cloud providers + an MCP
tool layer) and defaults to a local hearth engine. A tiny FastAPI server serves
the chat UI and proxies to the configured backend; that proxy is where the MCP
host and the tool-call loop will live.
"""

__version__ = "0.1.0"
