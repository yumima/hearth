"""Admin surface: health, readiness, hardware, backends, roles, model lifecycle.

These live under /admin so OpenAI SDK clients pointed at /v1 never see them.
Role bindings (view/rebind), model pull/remove, and the hardware probe all
hang here (design doc §1 admin endpoints, F5–F9).
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import CONTRACT_VERSION, __version__
from .. import config as cfgmod
from .. import hardware
from ..provision import Problem, Provisioner

router = APIRouter()


@router.get("/version")
async def version(_: Request):
    """Identity + contract version for consumer discovery / version-floor.

    `contract` is the stable consumer API version a client pins against;
    `version` is the engine build. `engine` is a human/grep-friendly id that
    also appears in the Server header, letting a client tell hearth apart from
    a bare Ollama on a nearby port.
    """
    return {
        "engine": f"hearth/{__version__}",
        "name": "hearth",
        "version": __version__,
        "contract": CONTRACT_VERSION,
    }


def _registry(request: Request) -> dict:
    return request.app.state.backends


def _cfg(request: Request) -> cfgmod.Config:
    return request.app.state.cfg


@router.post("/shutdown")
async def shutdown(request: Request):
    """End this gateway as `hearth stop` means it: in order, its Ollama with it.
    (A signal does that on Linux and macOS; Windows has none that reaches a
    process without a console.)"""
    end = request.app.state.shutdown
    if end is None:
        return JSONResponse({"error": "this gateway is not run by `hearth serve`"}, 409)
    end()
    return {"stopping": True}


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
            # `remote` tells a consumer (finterm's settings pane, `hearth status`)
            # that this role's prompts leave the machine — hearth is loopback-only
            # by default, so a cloud-bound role should never be invisible.
            name: {"model": r.model, "backend": r.backend,
                   "remote": cfg.role_is_remote(name)}
            for name, r in cfg.roles.items()
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
    # Optional: the CLI sends the context alongside the model so a hot rebind
    # and a config-file rebind agree. Absent → cleared, matching `hearth model use`.
    try:
        context = int(body.get("context") or 0)
    except (TypeError, ValueError):
        return JSONResponse({"error": "'context' must be an integer"}, 400)
    if context < 0:
        return JSONResponse({"error": "'context' must be >= 0"}, 400)
    cfg.roles[role] = cfgmod.RoleBinding(model=model, backend=backend, context=context)
    cfgmod.save(cfg)
    return {"role": role, "model": model, "backend": backend, "context": context,
            "remote": cfg.role_is_remote(role)}


# ── setting up: models and voices (provision.py) ──────────────────────────────


def _provisioner(request: Request) -> Provisioner:
    return request.app.state.provisioner


@router.get("/setup")
def setup_state(request: Request):  # in a thread: it asks the system and Ollama
    return _provisioner(request).state()


@router.post("/jobs")
async def submit_job(request: Request):
    body = await _object(request)
    try:
        job = _provisioner(request).submit(str(body.get("kind") or ""), str(body.get("what") or "").strip(),
                                           body.get("anyway") is True)
    except Problem as e:
        return JSONResponse({"error": str(e)}, 400)
    return job.as_dict()


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: int, request: Request):
    return {"stopping": _provisioner(request).stop_job(job_id)}


async def _object(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


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
def delete_model(model_id: str, request: Request):
    try:
        _provisioner(request).remove(model_id)
    except Problem as e:
        return JSONResponse({"error": str(e)}, 409)
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"Ollama did not answer: {e}"}, 502)
    return {"deleted": model_id}
