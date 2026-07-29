"""Search parsing and the SSRF guard. No network: parsing runs on a fixture.

The guard tests matter most — web_fetch is driven by a model, so "refuse
private addresses" is a boundary an LLM will probe by accident sooner or later.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hearth import websearch  # noqa: E402
from hearth.config import SearchConfig  # noqa: E402

# Trimmed from a real search.brave.com response: the semantic hooks the parser
# anchors on (data-type, search-snippet-title, line-clamp-dynamic) beside the
# hashed svelte class names it deliberately ignores.
_BRAVE_FIXTURE = """
<div class="snippet  svelte-jmfu5f" data-pos="0" data-type="web" data-keynav="true">
  <a href="https://example.com/a" class="svelte-14r20fy l1">
    <div class="site-name-content"><div class="text-ellipsis">Example</div></div>
    <div class="title search-snippet-title line-clamp-1 svelte-14r20fy">First Result Title</div>
  </a>
  <div class="content desktop-default-regular line-clamp-dynamic svelte-1cwdgg3">
    4 days ago - The first snippet body.
  </div>
</div>
<div class="snippet  svelte-jmfu5f" data-pos="1" data-type="web" data-keynav="true">
  <a href="https://example.org/b" class="svelte-14r20fy l1">
    <div class="title search-snippet-title line-clamp-1 svelte-14r20fy">Second &amp; Title</div>
  </a>
  <div class="content line-clamp-dynamic svelte-1cwdgg3">Jul 24, 2026 - Second body.</div>
</div>
"""


def _parse(html):
    """Exercise the block/title/description regexes the way _brave_html does."""
    starts = [m.start() for m in websearch._BRAVE_BLOCK.finditer(html)]
    out = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(html)
        block = html[start:end]
        m = websearch._HREF.search(block)
        tm = websearch._BRAVE_TITLE.search(block)
        dm = websearch._BRAVE_DESC.search(block)
        snippet = websearch._clean(websearch._strip_tags(dm.group(1)) if dm else "")
        age = ""
        am = websearch._AGE_PREFIX.match(snippet)
        if am:
            age, snippet = am.group(1), snippet[am.end():]
        out.append((websearch._clean(websearch._strip_tags(tm.group(1))), m.group(1), snippet, age))
    return out


def test_brave_html_parses_title_url_snippet():
    rows = _parse(_BRAVE_FIXTURE)
    assert len(rows) == 2
    assert rows[0][0] == "First Result Title"
    assert rows[0][1] == "https://example.com/a"
    assert rows[0][2] == "The first snippet body."


def test_brave_html_title_excludes_breadcrumb_chrome():
    """Falling back to the anchor text pulls in site name and breadcrumbs."""
    assert "Example" not in _parse(_BRAVE_FIXTURE)[0][0]


def test_brave_html_unescapes_entities():
    assert _parse(_BRAVE_FIXTURE)[1][0] == "Second & Title"


def test_age_prefix_is_split_into_its_own_field():
    rows = _parse(_BRAVE_FIXTURE)
    assert rows[0][3] == "4 days ago"
    assert rows[1][3] == "Jul 24, 2026"
    assert not rows[1][2].startswith("Jul 24")


def test_age_prefix_does_not_eat_ordinary_text():
    assert websearch._AGE_PREFIX.match("The 3 best ways - to cook") is None


# ── SSRF guard ────────────────────────────────────────────────────────────────

def test_loopback_blocked_by_default():
    assert websearch.host_is_blocked("127.0.0.1") is True
    assert websearch.host_is_blocked("localhost") is True


def test_loopback_allowed_under_explicit_opt_in():
    assert websearch.host_is_blocked("127.0.0.1", allow_local=True) is False


def test_cloud_metadata_blocked_even_with_opt_in():
    """169.254/16 hands out instance credentials; it is never a service the
    user meant to let the model debug."""
    assert websearch.host_is_blocked("169.254.169.254", allow_local=True) is True


def test_private_range_follows_the_opt_in():
    assert websearch.host_is_blocked("10.0.0.1") is True
    assert websearch.host_is_blocked("10.0.0.1", allow_local=True) is False


def test_unresolvable_and_empty_hosts_are_refused():
    assert websearch.host_is_blocked("") is True
    assert websearch.host_is_blocked("no-such-host.invalid") is True


# ── config ────────────────────────────────────────────────────────────────────

def test_brave_key_prefers_literal_then_env(monkeypatch):
    cfg = SearchConfig(brave_api_key="literal", brave_api_key_env="HEARTH_TEST_KEY")
    monkeypatch.setenv("HEARTH_TEST_KEY", "from-env")
    assert cfg.resolve_brave_key() == "literal"
    cfg.brave_api_key = ""
    assert cfg.resolve_brave_key() == "from-env"
    monkeypatch.delenv("HEARTH_TEST_KEY")
    assert cfg.resolve_brave_key() == ""


def test_first_line_trims_httpx_advice_footer():
    err = "Client error '429 Too Many Requests' for url 'https://x'\nFor more information check: https://mdn"
    assert websearch._first_line(err) == "Client error '429 Too Many Requests' for url 'https://x'"
