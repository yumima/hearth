"""Built-in MCP servers shipped with the client (launched as stdio subprocesses).

Each module here is runnable as ``python -m hearth_client.mcp_servers.<name>`` and speaks
MCP over stdio, so it can be wired into the client's ``mcp_servers`` config like any
external server.
"""
