"""The engine's own tools: web_search / web_fetch.

These are *server-side* tools in the hosted-provider sense — hearth injects the
schemas, the model calls them, and hearth executes them inside /v1 before the
response ever reaches the client. A consumer that knows nothing about search
(finterm, a bare OpenAI SDK script) gets live data for free; a consumer that
sends its own tools keeps them, since ours are appended, not substituted.

Naming matches Anthropic's server tools (``web_search``/``web_fetch``) because
models have seen that convention far more often than any name we'd invent.
"""

from __future__ import annotations

import datetime as _dt
import logging

from . import websearch

TOOL_NAMES = ("web_search", "web_fetch")

# Tool failures are reported to the *model* as text so it can recover, which
# means they never surface as an HTTP error. Log them too, or a systematically
# broken provider is invisible to the operator — the only symptom being the
# model saying it couldn't search.
_log = logging.getLogger("hearth.webtools")


def tool_specs(cfg) -> list[dict]:
    """OpenAI function schemas for the engine's tools."""
    fetch_desc = (
        "Fetch a web page or HTTP API by URL and return its readable text. Use "
        "this after web_search to read a promising result in full, or whenever "
        "the user gives you a URL. http/https only.")
    if cfg.allow_localhost:
        fetch_desc += (
            " Local addresses are permitted on this machine, so you can also "
            "read from services running here (e.g. http://127.0.0.1:8080/health) "
            "to diagnose them.")
    return [
        {"type": "function", "function": {
            "name": "web_search",
            "description":
                "Search the live web and return ranked results (title, URL, "
                "snippet, and date where available). Use this whenever the "
                "answer depends on current facts, recent events, releases, "
                "prices, versions, documentation, or anything that may have "
                "changed since training. Prefer searching over guessing.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string",
                          "description": "search query, phrased as you would type it"},
                "count": {"type": "integer",
                          "description": f"number of results (default {cfg.max_results}, max 10)"},
                "freshness": {"type": "string", "enum": ["pd", "pw", "pm", "py"],
                              "description": "restrict to the past day/week/month/year"},
            }, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "web_fetch",
            "description": fetch_desc,
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer",
                              "description": f"truncate the page (default {cfg.fetch_max_chars})"},
            }, "required": ["url"]}}},
    ]


def system_preamble(cfg) -> str:
    """A short system note appended when the tools are injected.

    Two jobs. First, the date: a local model's sense of "now" is its training
    cutoff, so without this it confidently answers stale questions instead of
    searching. Second, the policy — small models under-use tools unless told
    plainly when to reach for them.
    """
    today = _dt.date.today().isoformat()
    note = (
        f"Today's date is {today}. Your training data has a cutoff, so treat "
        "anything time-sensitive as potentially out of date.\n"
        "You have live internet access via the web_search and web_fetch tools. "
        "Use web_search whenever the question involves current events, recent "
        "releases or versions, prices, people's current roles, or any fact that "
        "could have changed — do not answer such questions from memory, and do "
        "not tell the user you cannot browse the web. After searching, call "
        "web_fetch on the most relevant result when the snippets are not enough. "
        "Cite the URLs you used in your answer.")
    if cfg.allow_localhost:
        note += ("\nweb_fetch may also reach services on this machine "
                 "(http://127.0.0.1:PORT/...), which you can use to inspect "
                 "local endpoints when diagnosing a problem.")
    return note


def _format_results(results: list[websearch.Result], query: str) -> str:
    lines = [f'Search results for "{query}":', ""]
    for i, r in enumerate(results, 1):
        age = f" ({r.age})" if r.age else ""
        lines.append(f"{i}. {r.title}{age}\n   {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
    lines.append("")
    lines.append("Call web_fetch on a URL above to read the full page.")
    return "\n".join(lines)


async def dispatch(name: str, args: dict, cfg) -> str:
    """Execute one engine tool and return the text the model will read.

    Errors come back as text rather than exceptions: a model that is told
    "search failed: rate limited" can say so or retry, whereas a 502 out of
    /v1 would abort a conversation that was otherwise fine.
    """
    try:
        if name == "web_search":
            count = min(10, max(1, int(args.get("count") or cfg.max_results)))
            results = await websearch.search(
                str(args.get("query") or ""), cfg, count,
                str(args.get("freshness") or ""))
            return _format_results(results, str(args.get("query") or ""))
        if name == "web_fetch":
            r = await websearch.fetch(str(args.get("url") or ""), cfg,
                                      int(args.get("max_chars") or 0))
            if "error" in r:
                return f"web_fetch error: {r['error']}"
            head = f"[{r['status']}] {r['url']} ({r['content_type']})\n\n"
            return head + r["text"] + ("\n\n…(truncated)" if r["truncated"] else "")
    except websearch.SearchError as e:
        _log.warning("web_search failed for %r: %s", args.get("query"), e)
        return (f"web_search failed: {e}\n"
                "Tell the user you could not reach a search provider rather "
                "than answering from memory.")
    except Exception as e:  # never let a tool bug kill the conversation
        _log.warning("%s failed: %s: %s", name, type(e).__name__, e, exc_info=True)
        return f"{name} failed: {type(e).__name__}: {e}"
    return f"unknown tool {name!r}"
