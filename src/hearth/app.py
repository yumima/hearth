"""FastAPI app factory.

Builds the backend-adapter registry from config at startup and closes it on
shutdown. The gateway holds no model state of its own — it resolves roles and
proxies to backends.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import backends as backends_mod
from . import config as cfgmod
from .middleware import SecurityMiddleware
from .routes import admin, v1, webui


def create_app(
    cfg: cfgmod.Config | None = None,
    *,
    allow_any_host: bool = False,
    api_key: str | None = None,
    backends_override: dict | None = None,
) -> FastAPI:
    """Build the gateway app.

    ``backends_override`` injects a pre-built adapter registry (used by tests
    with fakes); otherwise adapters are built from ``cfg.backends`` and closed
    on shutdown.
    """
    cfg = cfg or cfgmod.load()

    if backends_override is not None:
        registry = backends_override
        owned = False
    else:
        # httpx.AsyncClient construction doesn't need a running loop, so we can
        # build adapters eagerly. Doing it here (not in lifespan) means
        # app.state is populated even when the lifespan never runs — e.g. a
        # TestClient used without its `with` block.
        registry = {
            name: backends_mod.build(name, b.type, b.base_url)
            for name, b in cfg.backends.items()
        }
        owned = True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if owned:
                for b in registry.values():
                    await b.aclose()

    app = FastAPI(title="hearth", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.backends = registry
    app.add_middleware(
        SecurityMiddleware, allow_any_host=allow_any_host, api_key=api_key
    )
    app.include_router(v1.router, prefix="/v1")
    app.include_router(admin.router, prefix="/admin")
    app.include_router(webui.router)  # GET /app (chat UI) + / → /app
    return app
