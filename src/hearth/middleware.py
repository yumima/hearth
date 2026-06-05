"""Security middleware (design doc §4.7).

Single-user, single-host, loopback-bound. The practical threat is
DNS-rebinding: a malicious web page the user visits trying to reach
``localhost:11435`` from their browser. Mitigations:

- Require the ``Host`` header to be localhost/127.0.0.1 (reject anything
  else). A rebinding attack arrives with the attacker's hostname in Host.
- No permissive CORS (we never add CORS headers, so the browser blocks
  cross-origin reads).

Off-loopback binding requires explicit ``--unsafe-bind`` at the CLI, which
also forces an API key — that path sets ``allow_any_host`` and the bearer
check here.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__

_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}
_SERVER_HEADER = f"hearth/{__version__}"


def _host_only(value: str) -> str:
    # Strip the port; handle bracketed IPv6.
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    return value.rsplit(":", 1)[0] if ":" in value else value


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, allow_any_host: bool = False, api_key: str | None = None):
        super().__init__(app)
        self.allow_any_host = allow_any_host
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if not self.allow_any_host:
            host = _host_only(request.headers.get("host", ""))
            if host not in _ALLOWED_HOSTS:
                return JSONResponse(
                    {"error": {"message": "host not allowed", "type": "forbidden"}},
                    status_code=403,
                )
        if self.api_key is not None:
            auth = request.headers.get("authorization", "")
            token = auth[7:] if auth.lower().startswith("bearer ") else ""
            if token != self.api_key:
                return JSONResponse(
                    {"error": {"message": "invalid api key", "type": "unauthorized"}},
                    status_code=401,
                )
        response = await call_next(request)
        # Identify hearth to consumers (detection / version-floor); set on all
        # passed-through responses so even /v1/* carries it.
        response.headers["Server"] = _SERVER_HEADER
        return response
