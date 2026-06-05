"""Admin surface: health, readiness, hardware, backends, roles, model lifecycle.

These live under /admin so OpenAI SDK clients pointed at /v1 never see them.
Role bindings (view/rebind), model pull/remove, and the hardware probe all
hang here (design doc §1 admin endpoints, F5–F9).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import config as cfgmod
from .. import hardware

router = APIRouter()


def _registry(request: Request) -> dict:
    return request.app.state.backends


def _cfg(request: Request) -> cfgmod.Config:
    return request.app.state.cfg


@router.get("/health")
async def health(request: Request):
    backends = {}
    for name, b in _registry(request).items():
        backends[name] = await b.health()
    ok = all(backends.values()) if backends else False
    return JSONResponse(
        {"status": "ok" if ok else "degraded", "backends": backends},
        status_code=200 if ok else 503,
    )


@router.get("/ready")
async def ready(request: Request):
    """Ready when the primary_chat role resolves to a healthy backend."""
    cfg = _cfg(request)
    bound = cfg.backend_for_role("primary_chat")
    if not bound:
        return JSONResponse({"ready": False, "reason": "primary_chat unbound"}, 503)
    _, backend_cfg = bound
    backend = _registry(request).get(backend_cfg.name)
    healthy = bool(backend) and await backend.health()
    return JSONResponse({"ready": healthy}, status_code=200 if healthy else 503)


@router.get("/hardware")
async def hw(_: Request):
    return hardware.as_dict()


@router.get("/backends")
async def backends(request: Request):
    out = []
    for name, b in _registry(request).items():
        caps = b.capabilities
        out.append(
            {
                "name": name,
                "base_url": b.base_url,
                "healthy": await b.health(),
                "capabilities": {
                    "chat": caps.chat,
                    "embeddings": caps.embeddings,
                    "streaming": caps.streaming,
                    "tools": caps.tools,
                    "vision": caps.vision,
                    "json_mode": caps.json_mode,
                },
            }
        )
    return {"backends": out}


@router.get("/roles")
async def get_roles(request: Request):
    cfg = _cfg(request)
    return {
        "roles": {
            name: {"model": r.model, "backend": r.backend} for name, r in cfg.roles.items()
        }
    }


@router.put("/roles/{role}")
async def put_role(role: str, request: Request):
    """Hot-rebind a role without restart (F7). Persists to config.yaml."""
    cfg = _cfg(request)
    if role not in cfgmod.ROLE_NAMES:
        return JSONResponse({"error": f"unknown role {role!r}"}, 400)
    body = await request.json()
    model = body.get("model")
    backend = body.get("backend", "ollama")
    if not model:
        return JSONResponse({"error": "missing 'model'"}, 400)
    if backend not in cfg.backends:
        return JSONResponse({"error": f"unknown backend {backend!r}"}, 400)
    cfg.roles[role] = cfgmod.RoleBinding(model=model, backend=backend)
    cfgmod.save(cfg)
    return {"role": role, "model": model, "backend": backend}


@router.post("/models/pull")
async def pull_model(request: Request):
    body = await request.json()
    model = body.get("model")
    if not model:
        return JSONResponse({"error": "missing 'model'"}, 400)
    backend_name = body.get("backend", "ollama")
    backend = _registry(request).get(backend_name)
    if backend is None:
        return JSONResponse({"error": f"backend {backend_name!r} not running"}, 503)

    async def gen():
        async for line in backend.pull(model):
            yield line

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.delete("/models/{model_id:path}")
async def delete_model(model_id: str, request: Request):
    backend = _registry(request).get("ollama")
    if backend is None:
        return JSONResponse({"error": "ollama backend not running"}, 503)
    try:
        await backend.delete(model_id)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 502)
    return {"deleted": model_id}
