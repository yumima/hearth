"""Pure-function tests for the built-in MCP servers (no backend/network needed)."""

from __future__ import annotations

from mantel.mcp_servers import rag, web


def test_web_ssrf_blocks_internal_allows_public():
    for h in ["127.0.0.1", "localhost", "169.254.169.254", "192.168.1.1", "10.0.0.1", "::1", ""]:
        assert web._host_is_blocked(h) is True, h
    for h in ["8.8.8.8", "1.1.1.1"]:  # public literals — no DNS needed
        assert web._host_is_blocked(h) is False, h


def test_web_fetch_refuses_internal_and_bad_scheme():
    # Blocked hosts are refused before any connection is attempted.
    assert "refused" in web._do_fetch("http://127.0.0.1:11435/admin/version")["error"]
    assert "refused" in web._do_fetch("http://169.254.169.254/latest/meta-data/")["error"]
    assert "http(s)" in web._do_fetch("ftp://example.com/x")["error"]


def test_web_html_to_text_strips_scripts():
    assert web._html_to_text("<p>Hi <b>there</b></p><script>evil()</script>") == "Hi\nthere"


def test_rag_chunk_boundaries():
    assert rag._chunk("") == []
    assert rag._chunk("short") == ["short"]
    big = rag._chunk("word " * 600)            # well over CHUNK_CHARS
    assert len(big) >= 2
    assert all(c.strip() for c in big)         # no empty/whitespace chunks
