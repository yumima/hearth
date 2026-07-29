"""Live web access for the engine — search providers + page fetch.

Hearth's own answer to "the model only knows what it was trained on". This is
deliberately in the *engine*, not in a client: mantel, finterm, `hearth chat`
and `hearth code` all talk to /v1, so putting search here means every consumer
gets live data without shipping its own scraper.

Provider chain (config `search.provider: auto`), in order of preference:

1. **SearXNG** — a self-hosted metasearch instance (`search.searxng_url`).
   First because it is the only provider that reaches Google's index, and it
   does so without a key, an account, or a third party seeing the query: it
   runs on the user's own machine and aggregates the upstream engines itself.
   Google's own Custom Search JSON API is closed to new customers and retires
   on 2027-01-01, so this is the practical route to those results.
2. **Brave Search API** — the same source Anthropic licenses for Claude's
   web_search. Clean JSON, dated results, no scraping, and far steadier than
   any scrape — but it needs a (free-tier) key, read from an env var by default
   so it never lands in config.yaml or git.
3. **Brave HTML** — scrape search.brave.com. No key, works out of the box, but
   brittle by nature (a markup change breaks it) and rate-limited in practice —
   it starts answering 429 well inside a single agent session. DuckDuckGo is
   deliberately NOT in the chain: both its lite and html endpoints answer a bot
   challenge here, so it returns zero results.
4. **Wikipedia** — always-available last resort. Not a general web index, but
   for encyclopedic questions it beats returning nothing.

Fetch keeps the SSRF guard every LLM-driven fetcher needs (a model can be
talked into requesting http://169.254.169.254/…), with an explicit opt-in for
loopback so the agent can actually diagnose the services running on this box.
"""

from __future__ import annotations

import asyncio
import html as _html
import ipaddress
import json
import re
import socket
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote_plus, urlparse

import httpx

# A real browser UA. Brave's HTML endpoint serves a degraded/blocked page to an
# obvious bot string; this is the difference between 58 results and zero.
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/126.0.0.0 Safari/537.36")
_BRAVE_API = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_HTML = "https://search.brave.com/search"
_WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia's UA policy is enforced, not advisory: it 403s both a bare tool
# string ("hearth-search/0.1") and a browser string. What it accepts is a
# descriptive name plus a contact URL, so this identifies the project rather
# than impersonating a browser like the other providers require.
_WIKI_UA = "hearth/0.1 (local AI engine; https://github.com/yumima/hearth)"

_MAX_REDIRECTS = 5
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head",
              "nav", "footer", "header", "aside", "form"}


@dataclass
class Result:
    title: str
    url: str
    snippet: str = ""
    age: str = ""       # "2 days ago" / "2026-07-24" — providers that date results
    source: str = ""    # which provider produced it (shown to the model)

    def to_dict(self) -> dict:
        d = {"title": self.title, "url": self.url, "snippet": self.snippet}
        if self.age:
            d["age"] = self.age
        return d


class SearchError(RuntimeError):
    """No provider could answer. Carries the per-provider reasons for the model."""


# ── SSRF guard ────────────────────────────────────────────────────────────────

def host_is_blocked(host: str, allow_local: bool = False) -> bool:
    """Block hosts resolving to private/loopback/link-local/reserved space.

    On this box Ollama, hearth, mantel and finterm all listen on loopback, and
    cloud metadata sits at 169.254.169.254 — a model-driven fetch must not reach
    them by default. ``allow_local`` is the operator's explicit opt-in (config
    ``search.allow_localhost``) for the "let it debug my local services" case;
    even then link-local metadata stays blocked. Unresolvable → blocked.
    """
    if not host:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return True
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if addr.is_multicast or addr.is_unspecified:
            return True
        # 169.254.0.0/16 carries cloud instance-metadata credentials. It is
        # never a service the user meant to debug, so it stays blocked even
        # under allow_local.
        if addr.is_link_local:
            return True
        if addr.is_private or addr.is_loopback or addr.is_reserved:
            if not allow_local:
                return True
    return False


# ── HTML → text ───────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


def html_to_text(raw: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(raw)
    except Exception:
        return re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(p.parts)).strip()


def _strip_tags(fragment: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


def _clean(text: str, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


# ── providers ─────────────────────────────────────────────────────────────────

async def _brave_api(client: httpx.AsyncClient, query: str, count: int, key: str,
                     freshness: str = "") -> list[Result]:
    params: dict = {"q": query, "count": min(20, max(1, count)),
                    "text_decorations": 0, "result_filter": "web"}
    if freshness:
        params["freshness"] = freshness  # pd | pw | pm | py
    r = await client.get(_BRAVE_API, params=params, headers={
        "Accept": "application/json", "X-Subscription-Token": key})
    if r.status_code == 401:
        raise SearchError("brave api: key rejected (401)")
    if r.status_code == 429:
        raise SearchError("brave api: rate limited (429)")
    r.raise_for_status()
    out: list[Result] = []
    for item in ((r.json().get("web") or {}).get("results") or []):
        url = item.get("url") or ""
        if not url:
            continue
        out.append(Result(
            title=_clean(_strip_tags(item.get("title") or url), 200),
            url=url,
            snippet=_clean(_strip_tags(item.get("description") or "")),
            age=item.get("age") or item.get("page_age") or "",
            source="brave-api"))
    return out[:count]


async def _searxng(client: httpx.AsyncClient, query: str, count: int,
                   base: str) -> list[Result]:
    r = await client.get(f"{base.rstrip('/')}/search", params={
        "q": query, "format": "json", "categories": "general", "language": "en"})
    if r.status_code == 403:
        raise SearchError("searxng: JSON API disabled (add 'json' to "
                          "search.formats in settings.yml)")
    r.raise_for_status()
    out: list[Result] = []
    for item in (r.json().get("results") or []):
        url = item.get("url") or ""
        if not url:
            continue
        out.append(Result(
            title=_clean(item.get("title") or url, 200),
            url=url,
            snippet=_clean(item.get("content") or ""),
            age=(item.get("publishedDate") or "")[:10],
            source="searxng"))
    return out[:count]


# Brave renders each organic hit as a <div class="snippet …" data-type="web">
# containing a title div and a description div. Every anchor here is a
# *semantic* class or attribute (data-type, search-snippet-title,
# line-clamp-dynamic) rather than one of the hashed svelte-xxxxx names beside
# them, so a routine redeploy that rehashes the CSS doesn't break parsing.
_BRAVE_BLOCK = re.compile(
    r'<div[^>]*class="[^"]*\bsnippet\b[^"]*"[^>]*data-type="web"[^>]*>', re.I)
_BRAVE_TITLE = re.compile(
    r'<div[^>]*class="[^"]*search-snippet-title[^"]*"[^>]*>(.*?)</div>', re.I | re.S)
_BRAVE_DESC = re.compile(
    r'<div[^>]*class="[^"]*line-clamp-dynamic[^"]*"[^>]*>(.*?)</div>', re.I | re.S)
_HREF = re.compile(r'<a[^>]+href="(https?://[^"#]+)"[^>]*>(.*?)</a>', re.I | re.S)
# Brave prefixes a dated description with "4 days ago - " / "Jul 24, 2026 - ".
# Splitting it out gives the model the same title/snippet/age triple the API
# returns — and recency is exactly what it needs to judge a stale source.
_AGE_PREFIX = re.compile(
    r"^((?:\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago)"
    r"|(?:\w{3,9}\s+\d{1,2},\s+\d{4}))\s+-\s+", re.I)


async def _brave_html(client: httpx.AsyncClient, query: str, count: int) -> list[Result]:
    r = await client.get(_BRAVE_HTML, params={"q": query, "source": "web"},
                         headers={"User-Agent": _UA,
                                  "Accept-Language": "en-US,en;q=0.9"})
    r.raise_for_status()
    body = r.text
    starts = [m.start() for m in _BRAVE_BLOCK.finditer(body)]
    out: list[Result] = []
    seen: set[str] = set()
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else min(len(body), start + 8000)
        block = body[start:end]
        m = _HREF.search(block)
        if not m:
            continue
        url = _html.unescape(m.group(1))
        host = urlparse(url).hostname or ""
        if not host or host.endswith("brave.com") or url in seen:
            continue
        seen.add(url)
        tm = _BRAVE_TITLE.search(block)
        # Fall back to the anchor's own text if the title div moved: it carries
        # site name and breadcrumbs too, but a noisy result beats a dropped one.
        title = _clean(_strip_tags(tm.group(1) if tm else m.group(2)), 200)
        dm = _BRAVE_DESC.search(block)
        snippet = _clean(_strip_tags(dm.group(1)) if dm else "", 400)
        age = ""
        am = _AGE_PREFIX.match(snippet)
        if am:
            age, snippet = am.group(1), snippet[am.end():]
        out.append(Result(title=title or url, url=url, snippet=snippet,
                          age=age, source="brave-html"))
        if len(out) >= count:
            break
    if not out:
        raise SearchError("brave html: no results parsed (markup changed or "
                          "the request was challenged)")
    return out


async def _wikipedia(client: httpx.AsyncClient, query: str, count: int) -> list[Result]:
    r = await client.get(_WIKI_API, params={
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": min(10, count), "format": "json"},
        headers={"User-Agent": _WIKI_UA})
    r.raise_for_status()
    out: list[Result] = []
    for item in ((r.json().get("query") or {}).get("search") or []):
        title = item.get("title") or ""
        if not title:
            continue
        out.append(Result(
            title=title,
            url=f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
            snippet=_clean(_strip_tags(item.get("snippet") or "")),
            age=(item.get("timestamp") or "")[:10],
            source="wikipedia"))
    return out[:count]


# ── public API ────────────────────────────────────────────────────────────────

# Order the `auto` chain walks. A provider that isn't configured is skipped
# (with its reason recorded), so this order is a preference, not a requirement.
_CHAIN = ("searxng", "brave", "brave_html", "wikipedia")

# Short-lived result cache. An agent loop re-searches the same phrase often —
# refining a query, or a second round after a fetch — and the keyless providers
# rate-limit at agent speed (Brave's HTML endpoint starts 429ing well before a
# session is over). Caching identical queries is the cheapest way to stay under
# that ceiling without weakening results.
_CACHE_TTL = 600.0
_CACHE_MAX = 128
_cache: dict[tuple, tuple[float, list[Result]]] = {}


def _first_line(e: object) -> str:
    """Providers surface httpx errors that carry a trailing MDN advice line.
    Keep the first line: the rest is noise in a log and wasted tokens when the
    text is handed to a model."""
    return str(e).strip().splitlines()[0]


async def search(query: str, cfg, count: int = 0, freshness: str = "") -> list[Result]:
    """Run ``query`` through the configured provider chain.

    Returns the first provider's results that come back non-empty. Raises
    ``SearchError`` listing every provider's failure if none did — the model
    sees that text and can tell the user search is down rather than
    hallucinating an answer.
    """
    query = (query or "").strip()
    if not query:
        raise SearchError("empty query")
    count = count or cfg.max_results
    now = time.monotonic()
    ckey = (query.lower(), count, freshness, cfg.provider)
    hit = _cache.get(ckey)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    want = list(_CHAIN) if cfg.provider == "auto" else [cfg.provider]
    key = cfg.resolve_brave_key()
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0),
                                 follow_redirects=True) as client:
        for name in want:
            try:
                if name == "brave":
                    if not key:
                        errors.append("brave api: no key (set "
                                      f"${cfg.brave_api_key_env} or search.brave_api_key)")
                        continue
                    res = await _brave_api(client, query, count, key, freshness)
                elif name == "searxng":
                    if not cfg.searxng_url:
                        errors.append("searxng: not configured (search.searxng_url)")
                        continue
                    res = await _searxng(client, query, count, cfg.searxng_url)
                elif name == "brave_html":
                    res = await _brave_html(client, query, count)
                elif name == "wikipedia":
                    res = await _wikipedia(client, query, count)
                else:
                    errors.append(f"{name}: unknown provider")
                    continue
            except (SearchError, httpx.HTTPError, json.JSONDecodeError,
                    ValueError, asyncio.TimeoutError) as e:
                errors.append(f"{name}: {_first_line(e)}")
                continue
            if res:
                if len(_cache) >= _CACHE_MAX:  # crude bound; entries are small
                    _cache.clear()
                _cache[ckey] = (now, res)
                return res
            errors.append(f"{name}: no results")
    raise SearchError("; ".join(errors) or "no providers configured")


async def fetch(url: str, cfg, max_chars: int = 0) -> dict:
    """GET an http(s) URL and return its readable text.

    Redirects are followed manually so *every* hop is re-validated: a public URL
    is free to 30x to http://127.0.0.1/… and httpx's own redirect handling would
    sail straight past the guard.
    """
    max_chars = max_chars or cfg.fetch_max_chars
    target = (url or "").strip()
    if not re.match(r"^https?://", target, re.I):
        # Bare host:port is what a model reaches for when asked to poke a local
        # service; treat it as http rather than erroring on a technicality.
        if re.match(r"^[\w.-]+(:\d+)?(/|$)", target):
            target = "http://" + target
        else:
            return {"error": "only http(s) URLs are supported"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0),
                                 follow_redirects=False,
                                 headers={"User-Agent": _UA}) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            host = urlparse(target).hostname or ""
            if host_is_blocked(host, cfg.allow_localhost):
                hint = ("" if cfg.allow_localhost else
                        " (set search.allow_localhost: true to let the agent "
                        "reach services on this machine)")
                return {"error": f"refused: {host!r} resolves to a private/loopback/"
                                 f"reserved address{hint}"}
            try:
                r = await client.get(target)
            except httpx.HTTPError as e:
                return {"error": f"fetch failed: {e}"}
            if r.is_redirect and r.headers.get("location"):
                target = str(r.url.join(r.headers["location"]))
                continue
            break
        else:
            return {"error": "too many redirects"}
    ctype = r.headers.get("content-type", "")
    text = html_to_text(r.text) if "html" in ctype.lower() else r.text.strip()
    return {"status": r.status_code, "url": str(r.url), "content_type": ctype,
            "text": text[: max(1, max_chars)], "truncated": len(text) > max_chars}
