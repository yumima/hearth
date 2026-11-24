"""Security middleware (design doc §4.7).

Single-user, single-host, loopback-bound. The practical threat is
DNS-rebinding: a malicious web page the user visits trying to reach
``localhost:11435`` from their browser. Mitigations:

- Require the ``Host`` header to be localhost/127.0.0.1 (reject anything
  else). A rebinding attack arrives with the attacker's hostname in Host.
- No permissive CORS (we never add CORS headers, so the browser blocks
  cross-origin reads).
- Refuse a request another web page sent. CORS does not stop it from being
  SENT: a page may POST here without a preflight (a "simple" request, whatever
  JSON its body holds) and start a download or run the model, unable only to
  read the answer. A browser names the page's origin on such a request, and
  says "cross-site" on one for an image or a script; only our own page passes,
  and a person following a link here. Clients that are not web pages send
  neither. (So a page another server serves — a proxy under another name, a
  browser extension — cannot use hearth; such a page needs its own server side.)

Off-loopback binding requires explicit ``--unsafe-bind`` at the CLI, which
also forces an API key — that path sets ``allow_any_host`` and the bearer
check here.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from . import __version__

_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}
_SERVER_HEADER = f"hearth/{__version__}"


def _same_origin(origin: str, host: str) -> bool:
    """Is `Origin` the page this server itself served (as `Host` names it)?
    The scheme is left out: a TLS proxy in front changes it and nothing else.
    "null" (a sandboxed frame, a file) has no host, so it never matches."""
    try:
        return bool(host) and urlsplit(origin).netloc.lower() == host.lower()
    except ValueError:
        return False


def _host_only(value: str) -> str:
    # Strip the port; handle bracketed IPv6.
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    return value.rsplit(":", 1)[0] if ":" in value else value


class SecurityMiddleware:
    """Pure ASGI, not BaseHTTPMiddleware: that wrapper hides the client's
    disconnect from the route (request.is_disconnected() never turns true),
    so a request whose client hung up — a stopped read-aloud — kept the
    engine synthesising for nobody while the next request queued behind it."""

    def __init__(self, app, *, allow_any_host: bool = False, api_key: str | None = None):
        self.app = app
        self.allow_any_host = allow_any_host
        self.api_key = api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if not self.allow_any_host:
            host = _host_only(headers.get("host", ""))
            if host not in _ALLOWED_HOSTS:
                await self._reject(scope, receive, send, 403, "host not allowed", "forbidden")
                return
        origins = headers.getlist("origin")
        followed_link = (scope["method"] == "GET" and headers.get("sec-fetch-mode") == "navigate"
                         and headers.get("sec-fetch-dest") == "document")
        if any(not _same_origin(o, headers.get("host", "")) for o in origins) or (
                headers.get("sec-fetch-site") in ("cross-site", "same-site") and not followed_link):
            await self._reject(scope, receive, send, 403, "cross-origin request refused", "forbidden")
            return
        if self.api_key is not None:
            auth = headers.get("authorization", "")
            token = auth[7:] if auth.lower().startswith("bearer ") else ""
            if token != self.api_key:
                await self._reject(scope, receive, send, 401, "invalid api key", "unauthorized")
                return

        async def send_with_server(message):
            # Identify hearth to consumers (detection / version-floor); set on
            # all passed-through responses so even /v1/* carries it.
            if message["type"] == "http.response.start":
                raw = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"server"]
                raw.append((b"server", _SERVER_HEADER.encode()))
                message = {**message, "headers": raw}
            await send(message)

        await self.app(scope, receive, send_with_server)

    @staticmethod
    async def _reject(scope, receive, send, status: int, message: str, etype: str) -> None:
        response = JSONResponse({"error": {"message": message, "type": etype}}, status_code=status)
        await response(scope, receive, send)
