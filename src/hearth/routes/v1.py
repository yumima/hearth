"""OpenAI-compatible surface: /v1/chat/completions, /v1/embeddings, /v1/models.

The gateway resolves the request's ``model`` (role alias or literal id) to a
concrete ``(model_id, backend)``, checks the backend advertises the needed
capability, then proxies. Streaming is a transparent SSE pass-through so the
chunk format stays byte-for-byte OpenAI.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import ROLE_NAMES, Config

router = APIRouter()


def _err(status: int, message: str, etype: str = "invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": etype}}, status_code=status)


def _registry(request: Request) -> dict:
    return request.app.state.backends


def _cfg(request: Request) -> Config:
    return request.app.state.cfg


@router.get("/models")
async def list_models(request: Request):
    """List servable models in OpenAI shape (F4).

    Concrete backend models plus the role aliases — so a consumer's model
    dropdown can pick either ``primary_chat`` or ``qwen2.5:14b...``. Role
    *bindings* (which alias → which model) live at /admin/roles so OpenAI SDK
    clients don't see non-standard fields here.
    """
    cfg = _cfg(request)
    data: list[dict] = []
    seen: set[str] = set()
    for backend in _registry(request).values():
        try:
            for m in await backend.list_models():
                if m.id and m.id not in seen:
                    seen.add(m.id)
                    data.append(
                        {"id": m.id, "object": "model", "created": 0, "owned_by": backend.name}
                    )
        except Exception:
            continue  # a down backend shouldn't blank the whole list
    for role in ROLE_NAMES:
        if role in cfg.roles and role not in seen:
            data.append({"id": role, "object": "model", "created": 0, "owned_by": "hearth-role"})
    return {"object": "list", "data": data}


@router.post("/chat/completions")
async def chat_completions(request: Request):
    cfg = _cfg(request)
    payload = await request.json()
    model = payload.get("model")
    if not model:
        return _err(400, "missing 'model'")
    try:
        concrete, backend_cfg = cfg.resolve(model)
    except KeyError:
        return _err(400, f"role {model!r} is not bound")
    except LookupError as e:
        return _err(503, str(e), "backend_unavailable")
    backend = _registry(request).get(backend_cfg.name)
    if backend is None:
        return _err(503, f"backend {backend_cfg.name!r} not running", "backend_unavailable")

    # Capability gate (C3): refuse rather than pass an unsupported request down.
    if payload.get("tools") and not backend.capabilities.tools:
        return _err(400, f"model {concrete!r} on backend {backend.name!r} does not support tools")
    if payload.get("stream") and not backend.capabilities.streaming:
        return _err(400, f"model {concrete!r} does not support streaming")

    payload = {**payload, "model": concrete}

    if payload.get("stream"):
        async def gen():
            try:
                async for chunk in backend.chat_stream(payload):
                    yield chunk
            except Exception as e:  # surface as a final SSE error frame
                import json

                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        return await backend.chat(payload)
    except Exception as e:
        return _err(502, f"backend error: {e}", "backend_error")


@router.post("/embeddings")
async def embeddings(request: Request):
    cfg = _cfg(request)
    payload = await request.json()
    model = payload.get("model") or "embedding"  # default to the embedding role
    try:
        concrete, backend_cfg = cfg.resolve(model)
    except KeyError:
        return _err(400, f"role {model!r} is not bound")
    except LookupError as e:
        return _err(503, str(e), "backend_unavailable")
    backend = _registry(request).get(backend_cfg.name)
    if backend is None:
        return _err(503, f"backend {backend_cfg.name!r} not running", "backend_unavailable")
    if not backend.capabilities.embeddings:
        return _err(400, f"backend {backend.name!r} does not serve embeddings")
    try:
        return await backend.embeddings({**payload, "model": concrete})
    except Exception as e:
        return _err(502, f"backend error: {e}", "backend_error")
