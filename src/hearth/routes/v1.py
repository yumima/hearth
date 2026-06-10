"""OpenAI-compatible surface: /v1/chat/completions, /v1/embeddings, /v1/models.

The gateway resolves the request's ``model`` (role alias or literal id) to a
concrete ``(model_id, backend)``, checks the backend advertises the needed
capability, then proxies. Streaming is a transparent SSE pass-through so the
chunk format stays byte-for-byte OpenAI.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

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

    # `think` is a hearth extension (not an OpenAI field): toggle a reasoning
    # model's chain-of-thought. It only works over the backend's native API, so
    # pop it from the OpenAI payload and route to the native path when present.
    # Absent (the common case) → byte-for-byte OpenAI passthrough, unchanged.
    think = payload.pop("think", None)
    use_native = think is not None and hasattr(backend, "chat_stream_native")

    payload = {**payload, "model": concrete}

    if payload.get("stream"):
        async def gen():
            try:
                source = (backend.chat_stream_native(payload, bool(think)) if use_native
                          else backend.chat_stream(payload))
                async for chunk in source:
                    yield chunk
            except Exception as e:  # surface as a final SSE error frame
                import json

                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        if use_native:
            return await backend.chat_native(payload, bool(think))
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


# ── Text-to-speech (Piper) ────────────────────────────────────────────────────
# Not an Ollama capability — Piper is a standalone local engine. Exposed in the
# OpenAI /v1/audio/speech shape so any client (incl. mantel) works unmodified.
# Voices are provisioned by `hearth voice <id>` into ~/.local/share/finterm/piper.


def _piper_bin() -> str | None:
    return shutil.which("piper") or shutil.which("piper-tts")


def _voice_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "finterm" / "piper"


def _find_voice(voice: str | None) -> Path | None:
    """Resolve a Piper voice id (e.g. 'en_US-amy-medium') to its .onnx, or the
    first provisioned voice when none is requested."""
    d = _voice_dir()
    if voice:
        name = os.path.basename(voice)  # basename only — never escape the voice dir
        if name:
            p = d / (name if name.endswith(".onnx") else f"{name}.onnx")
            if p.exists():
                return p
    onnx = sorted(d.glob("*.onnx"))
    return onnx[0] if onnx else None


@router.post("/audio/speech")
async def audio_speech(request: Request):
    """Synthesize speech with Piper (OpenAI /v1/audio/speech shape) → WAV bytes."""
    payload = await request.json()
    text = (payload.get("input") or "").strip()
    if not text:
        return _err(400, "missing 'input'")
    text = text[:8000]  # bound synthesis time/memory for direct callers
    piper = _piper_bin()
    if piper is None:
        return _err(503, "piper not installed — run `hearth voice <id>` to provision TTS",
                    "backend_unavailable")
    voice = _find_voice(payload.get("voice"))
    if voice is None:
        return _err(503, "no Piper voice provisioned — run `hearth voice en_US-amy-medium`",
                    "backend_unavailable")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    tmp.close()
    try:
        proc = await asyncio.create_subprocess_exec(
            piper, "--model", str(voice), "--output_file", out_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate(text.encode("utf-8"))
        if proc.returncode != 0 or not os.path.getsize(out_path):
            return _err(502, f"piper failed: {err.decode('utf-8', 'ignore')[:200]}", "backend_error")
        data = Path(out_path).read_bytes()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    return Response(content=data, media_type="audio/wav", headers={"Cache-Control": "no-store"})


# ── Speech-to-text (faster-whisper) ───────────────────────────────────────────
# Standalone local engine (not Ollama). OpenAI /v1/audio/transcriptions shape.
# The model (default "base") is loaded once and cached; it downloads on first use.

_whisper_cache: dict = {}
# Allowlisted faster-whisper sizes. A non-size string is treated by faster-whisper
# as a HuggingFace repo id (outbound fetch) or a local path, so reject anything else.
_WHISPER_SIZES = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium",
    "medium.en", "large-v1", "large-v2", "large-v3", "large", "distil-large-v3",
}


def _get_whisper(size: str):
    model = _whisper_cache.get(size)
    if model is None:
        from faster_whisper import WhisperModel  # heavy import — defer to first use
        model = WhisperModel(size, device="cpu", compute_type="int8")
        _whisper_cache[size] = model
    return model


@router.post("/audio/transcriptions")
async def audio_transcriptions(request: Request):
    """Transcribe audio with faster-whisper (OpenAI /v1/audio/transcriptions shape)."""
    try:
        form = await request.form()
    except Exception:
        return _err(400, "expected multipart/form-data with a 'file' field")
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return _err(400, "missing audio 'file'")
    size = (form.get("model") or "").strip()
    if size in ("", "whisper-1", "whisper"):  # OpenAI clients send 'whisper-1'
        size = os.environ.get("HEARTH_WHISPER_MODEL", "base")
    if size not in _WHISPER_SIZES:  # reject arbitrary HF ids / local paths
        return _err(400, f"unknown whisper model {size!r}")
    suffix = Path(getattr(upload, "filename", "") or "audio.wav").suffix or ".wav"
    data = await upload.read()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()

    def _run() -> str:
        model = _get_whisper(size)  # may download the model on first call
        segments, _info = model.transcribe(tmp.name, beam_size=5)
        return "".join(seg.text for seg in segments).strip()

    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _run)
    except Exception as e:
        return _err(502, f"transcription failed: {e}", "backend_error")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return JSONResponse({"text": text})
