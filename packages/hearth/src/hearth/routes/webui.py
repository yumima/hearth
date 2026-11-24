"""Serve the local chat UI at GET /app.

The UI is plain HTML/CSS/JS bundled in the package (``webui/index.html``).
Serving it from the gateway itself means it is SAME-ORIGIN with the API
(``http://127.0.0.1:11435``): the browser/webview's streaming ``fetch`` to
``/v1/chat/completions`` works with no CORS, and the localhost-only Host check
in SecurityMiddleware passes. There is no second web server and nothing to
configure — ``hearth start`` already serves it, and ``hearth client`` opens a
desktop window pointed at it.
"""

from __future__ import annotations

from importlib import resources

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()


def _index_html() -> str | None:
    try:
        return resources.files("hearth").joinpath("webui/index.html").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None


@router.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def chat_app(request: Request) -> HTMLResponse:
    html = _index_html()
    if html is None:
        return HTMLResponse(
            "<h1>Hearth</h1><p>Chat UI asset is missing from this install.</p>",
            status_code=500,
        )
    asks = "true" if request.app.state.cfg.search.chat else "false"
    return HTMLResponse(html.replace("__WEB_SEARCH__", asks), headers={
        # Not inside another site's page: what is typed there would be ours.
        "X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'"})


@router.get("/", include_in_schema=False)
async def root_to_app() -> RedirectResponse:
    # A bare visit to the gateway lands on the chat UI instead of a 404.
    return RedirectResponse(url="/app")
