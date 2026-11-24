"""The setup page, at / — served by the gateway, so it is same-origin with the API
and inherits the localhost-only security posture."""

from __future__ import annotations

from importlib import resources

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()

_HEADERS = {
    # Not inside another site's page: what is clicked there would be done here.
    "X-Frame-Options": "DENY",
    # Its own code and styles and this server, nothing else.
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                               "connect-src 'self'; img-src data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def setup_page() -> HTMLResponse:
    html = resources.files("hearth").joinpath("webui/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers=_HEADERS)


@router.get("/app", include_in_schema=False)
async def old_chat_page() -> RedirectResponse:
    return RedirectResponse(url="/")  # chat is mantel's now
