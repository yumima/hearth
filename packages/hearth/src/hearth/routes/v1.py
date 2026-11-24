"""OpenAI-compatible surface: /v1/chat/completions, /v1/embeddings, /v1/models.

The gateway resolves the request's ``model`` (role alias or literal id) to a
concrete ``(model_id, backend)``, checks the backend advertises the needed
capability, then proxies. Streaming is a transparent SSE pass-through so the
chunk format stays byte-for-byte OpenAI.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from .. import comfy_supervisor, kokoro_tts, ollama_supervisor, paths, speech_text, toolloop, tts, webtools
from ..config import ROLE_NAMES, Config

router = APIRouter()
log = logging.getLogger("hearth.tts")

# Rough English/code average; good enough to keep a tool result from evicting
# the conversation, which is all this estimate is used for.
_CHARS_PER_TOKEN = 4
# Share of the context window a single fetched page may occupy.
_TOOL_RESULT_SHARE = 0.4


def _with_preamble(messages: list, note: str) -> list:
    """Fold the engine's tool note into the conversation's system prompt.

    Appended to the caller's own system message rather than prepended as a
    second one: consumers like finterm spend real effort on their system
    prompt, and a competing system turn ahead of it measurably weakens how
    well small models follow it.
    """
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str):
                out[i] = {**m, "content": content.rstrip() + "\n\n" + note}
                return out
            break
    return [{"role": "system", "content": note}] + out


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
    dropdown can pick either ``primary_chat`` or ``qwen3.8:27b``. Role
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
    # `web_search` is likewise a hearth extension: only a request that says
    # True gets the engine's web tools (see SearchConfig).
    want_search = payload.pop("web_search", None) is True

    # A configured per-role context window also needs the native path — see
    # OllamaBackend._to_native. Both extensions land on the same route.
    ctx = cfg.context_for(model)
    use_native = (think is not None or bool(ctx)) and hasattr(backend, "chat_stream_native")

    payload = {**payload, "model": concrete}
    if use_native and ctx:
        payload["num_ctx"] = ctx

    scfg = cfg.search
    search_on = bool(
        want_search
        and scfg.enabled
        and backend.capabilities.tools
        and payload.get("tool_choice") != "none"
    )
    if search_on:
        payload["tools"] = list(payload.get("tools") or []) + webtools.tool_specs(scfg)
        payload["messages"] = _with_preamble(payload.get("messages") or [],
                                             webtools.system_preamble(scfg))
        # Size a fetched page against the context that actually exists. Ollama
        # answers an over-long request by silently discarding the oldest
        # messages — the system prompt and the user's question — so a 20k-char
        # page against an 8k window doesn't fail loudly, it just makes the model
        # look like it ignored the task. Reserve most of the window for the
        # conversation and let one tool result have the rest.
        ctx_tokens = ctx or ollama_supervisor.effective_default_context()
        budget = max(2000, int(ctx_tokens * _CHARS_PER_TOKEN * _TOOL_RESULT_SHARE))
        if scfg.fetch_max_chars > budget:
            scfg = replace(scfg, fetch_max_chars=budget)

    async def _dispatch(name: str, args: dict) -> str:
        return await webtools.dispatch(name, args, scfg)

    if payload.get("stream"):
        def stream_fn(body: dict):
            return (backend.chat_stream_native(body, bool(think)) if use_native
                    else backend.chat_stream(body))

        async def gen():
            try:
                source = (
                    toolloop.stream_with_tools(
                        stream_fn, payload, webtools.TOOL_NAMES, _dispatch,
                        scfg.max_tool_rounds, scfg.progress)
                    if search_on else stream_fn(payload))
                async for chunk in source:
                    yield chunk
            except Exception as e:  # surface as a final SSE error frame
                import json

                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def chat_fn(body: dict) -> dict:
        if use_native:
            return await backend.chat_native(body, bool(think))
        return await backend.chat(body)

    try:
        work = (toolloop.chat_with_tools(
                    chat_fn, payload, webtools.TOOL_NAMES, _dispatch, scfg.max_tool_rounds)
                if search_on else chat_fn(payload))
        # A non-streaming request is silent until it finishes. Without this, one
        # whose client hung up keeps the model busy and everyone else queues.
        return await _unless_disconnected(request, work)
    except _ClientGone:
        return Response(status_code=499)
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
# Voices are provisioned by `hearth voice <id>` into paths.voices_dir("piper").


def _piper_bin() -> str | None:
    # Prefer PATH (covers user installs and the `piper-tts` pip package).
    found = shutil.which("piper") or shutil.which("piper-tts")
    if found:
        return found
    # Fall back to the canonical `hearth voice` install location. This is what
    # lets TTS work under systemd --user, whose bare PATH omits ~/.local/bin
    # (where `hearth voice` symlinks piper) — without it, a service install can
    # provision a voice yet still 503 with "piper not installed". Mirrors the
    # extract target in cli.py:_ensure_piper().
    cand = Path.home() / ".local" / "share" / "piper" / "piper" / "piper"
    if cand.exists() and os.access(cand, os.X_OK):
        return str(cand)
    return None


def _voices() -> dict[str, Path]:
    """Piper voices by id, hearth's own folder first (see paths.voice_dirs)."""
    out: dict[str, Path] = {}
    for d in paths.voice_dirs("piper"):
        for p in sorted(d.glob("*.onnx")) if d.is_dir() else []:
            out.setdefault(p.stem, p)
    return dict(sorted(out.items()))


def _find_voice(voice: str | None) -> Path | None:
    """Resolve a Piper voice id (e.g. 'en_US-amy-medium') to its .onnx. With no
    voice requested, default to English — NOT whatever sorts first alphabetically
    (that was 'de_DE-…', so default read-aloud came out German)."""
    have = _voices()
    if isinstance(voice, str) and voice:  # JSON may carry any type here
        name = voice.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".onnx")
        if name in have:
            return have[name]
    default = os.environ.get("HEARTH_TTS_VOICE", "en_US-amy-medium")
    if default in have:
        return have[default]
    english = [p for v, p in have.items() if v.startswith("en_")]  # any English voice next
    return (english or list(have.values()) or [None])[0]


class _ClientGone(Exception):
    pass


async def _unless_disconnected(request: Request, coro):
    """Await `coro`, cancelling it if the client hangs up first. Starlette does
    not cancel a handler on disconnect, so without this a stopped read-aloud
    keeps the engine busy synthesising audio nobody will hear — and the next
    request queues behind it (Kokoro holds one lock for the whole text)."""
    task = asyncio.ensure_future(coro)
    while True:
        done, _ = await asyncio.wait({task}, timeout=0.25)
        if done:
            return task.result()
        if await request.is_disconnected():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise _ClientGone()


def _wants_kokoro(payload: dict) -> bool:
    """`model: "kokoro"`, or a Kokoro-shaped voice id ('af_heart') with no
    model named. `model: "piper"` (and any OpenAI model name) keeps Piper."""
    model = str(payload.get("model") or "").lower()
    if model == "kokoro":
        return True
    if model:
        return False
    return kokoro_tts.engine.is_voice_id(payload.get("voice"))


@router.post("/audio/speech")
async def audio_speech(request: Request):
    """Synthesize speech (OpenAI /v1/audio/speech shape) → WAV bytes.

    Kokoro when asked for (natural voices); Piper otherwise, and as the
    fallback when Kokoro isn't installed or fails — read-aloud keeps working,
    just in the plainer voice. `X-Hearth-TTS-Engine` says which one answered."""
    payload = await request.json()
    raw = payload.get("input") or ""
    text = raw.strip() if isinstance(raw, str) else ""
    if not text:
        return _err(400, "missing 'input'")
    text = text[:8000]  # bound synthesis time/memory for direct callers
    # Trailing line breaks are meaningful to Kokoro: the chunk ends a paragraph
    # or a section, so it owes that pause at its end (kokoro_tts.trailing_break).
    text += "\n" * kokoro_tts.trailing_break(raw)
    headers = {"Cache-Control": "no-store"}
    if _wants_kokoro(payload):
        try:
            speed = float(payload.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        try:
            data = await _unless_disconnected(request, kokoro_tts.engine.synth(text, payload.get("voice"), speed))
            return Response(content=data, media_type="audio/wav",
                            headers={**headers, "X-Hearth-TTS-Engine": "kokoro"})
        except _ClientGone:
            return Response(status_code=499)  # nobody is listening; nothing to send
        except kokoro_tts.KokoroError as e:
            log.warning("kokoro unavailable, falling back to piper: %s", e)
        payload = {**payload, "voice": None}  # a Kokoro id means nothing to Piper
    piper = _piper_bin()
    if piper is None:
        return _err(503, "piper not installed — run `hearth voice <id>` to provision TTS",
                    "backend_unavailable")
    voice = _find_voice(payload.get("voice"))
    if voice is None:
        return _err(503, "no Piper voice provisioned — run `hearth voice en_US-amy-medium`",
                    "backend_unavailable")
    try:
        # Piper phonemizes with espeak too: numbers and money as words first.
        data = await _unless_disconnected(request, tts.pool.synth(piper, voice, speech_text.normalize(text)))
    except _ClientGone:
        return Response(status_code=499)
    except tts.TTSError as e:
        return _err(502, str(e), "backend_error")
    return Response(content=data, media_type="audio/wav",
                    headers={**headers, "X-Hearth-TTS-Engine": "piper"})


@router.get("/audio/voices")
async def audio_voices():
    """The voices /audio/speech can use right now (hearth extension, not OpenAI).
    `model` is what to send back in the speech request; `default` is the one
    a client should pick when the user hasn't."""
    voices: list[dict] = []
    default = None
    if kokoro_tts.installed():
        for vid, label in kokoro_tts.CURATED:
            voices.append({"id": vid, "model": "kokoro", "label": label})
        # What /audio/speech uses for a Kokoro request naming no voice.
        default = {"model": "kokoro", "id": kokoro_tts.DEFAULT_VOICE}
    for p in _voices().values():
        if p.name.startswith("en_"):  # read-aloud speaks English text
            voices.append({"id": p.stem, "model": "piper",
                           "label": f"{p.stem.split('-')[1].capitalize()} — Piper, fast ({p.stem.split('-')[0]})"
                           if p.stem.count("-") >= 2 else p.stem})
    if default is None:
        piper_default = _find_voice(None)  # same resolution as a voice-less speech request
        if piper_default is not None:
            default = {"model": "piper", "id": piper_default.stem}
    return {"voices": voices, "default": default}


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
        # Default to CPU (reliable). GPU is opt-in via HEARTH_WHISPER_DEVICE=cuda —
        # it needs ctranslate2's CUDA-12 libs (libcublas.so.12 / cuDNN), which
        # aren't present on a CUDA-13 box; without them CUDA *constructs* fine but
        # fails at transcribe. CPU int8 is a few seconds for a short clip — fine
        # for voice. Construct-time CUDA errors fall back to CPU here; a transcribe
        # failure drops the model from cache (see audio_transcriptions).
        device = os.environ.get("HEARTH_WHISPER_DEVICE", "cpu")
        try:
            model = WhisperModel(size, device=device,
                                 compute_type="float16" if device == "cuda" else "int8")
        except Exception:
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
    # Optional ISO language hint (e.g. 'it', 'fr', 'zh') — sharply improves accuracy
    # for a learner speaking a non-English language; None = whisper auto-detect.
    language = (form.get("language") or "").strip() or None
    suffix = Path(getattr(upload, "filename", "") or "audio.wav").suffix or ".wav"
    data = await upload.read()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()

    def _run() -> str:
        model = _get_whisper(size)  # may download the model on first call
        # vad_filter strips non-speech (whisper hallucinates repeated phantom text
        # on silence/noise); condition_on_previous_text=False stops repeat loops.
        segments, _info = model.transcribe(
            tmp.name, beam_size=5, language=language,
            vad_filter=True, condition_on_previous_text=False)
        return "".join(seg.text for seg in segments).strip()

    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _run)
    except Exception as e:
        _whisper_cache.pop(size, None)  # don't keep a broken model cached
        return _err(502, f"transcription failed: {e}", "backend_error")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return JSONResponse({"text": text})


# ── Image generation (ComfyUI / SDXL) ─────────────────────────────────────────
# Standalone local engine. OpenAI /v1/images/generations shape (returns b64_json).
# Drives a running ComfyUI (HTTP API on :8188) with a minimal SDXL txt2img graph.

_COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
_SDXL_CKPT = os.environ.get("HEARTH_SDXL_CKPT", "sd_xl_base_1.0.safetensors")
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


async def _free_ollama_vram(client: httpx.AsyncClient) -> None:
    """Best-effort: unload Ollama's resident models so the image model has VRAM.
    On a 12 GB card SDXL can't coexist with a 14B chat model; the next chat
    request transparently reloads it (a few seconds). Never fail image-gen on this."""
    try:
        ps = (await client.get(f"{_OLLAMA_URL}/api/ps", timeout=5.0)).json()
    except Exception:
        return
    for m in ps.get("models", []):
        name = m.get("name") or m.get("model")
        if not name:
            continue
        try:
            await client.post(f"{_OLLAMA_URL}/api/generate",
                              json={"model": name, "keep_alive": 0}, timeout=10.0)
        except Exception:
            pass


async def _warm_chat(cfg) -> None:
    """Background: reload the chat model into the VRAM just freed by image-gen, so
    the user's next message isn't a ~15-20s cold start. Best-effort."""
    try:
        concrete, _backend = cfg.resolve("primary_chat")
    except Exception:
        return
    keep = os.environ.get("HEARTH_KEEP_ALIVE", "30m")
    body: dict = {"model": concrete, "keep_alive": keep}
    # Warm at the context chats will ask for: Ollama reloads a model whose
    # num_ctx changes, so warming at its default just buys a second cold start.
    if ctx := cfg.context_for("primary_chat"):
        body["options"] = {"num_ctx": ctx}
    try:
        async with httpx.AsyncClient(timeout=180.0) as c:
            await c.post(f"{_OLLAMA_URL}/api/generate", json=body)
    except Exception:
        pass


def _sdxl_workflow(prompt: str, negative: str, w: int, h: int,
                   steps: int, cfg: float, seed: int) -> dict:
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": _SDXL_CKPT}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "mantel", "images": ["8", 0]}},
    }


@router.post("/images/generations")
async def images_generations(request: Request):
    """Generate an image with ComfyUI/SDXL (OpenAI /v1/images/generations shape)."""
    payload = await request.json()
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return _err(400, "missing 'prompt'")
    prompt = prompt[:2000]
    try:
        w, h = (int(x) for x in str(payload.get("size") or "1024x1024").lower().split("x"))
    except Exception:
        w, h = 1024, 1024
    w = max(256, min(1536, w)); h = max(256, min(1536, h))
    steps = max(1, min(60, int(payload.get("steps") or 25)))
    cfg = float(payload.get("cfg") or 7.0)
    seed = int(payload.get("seed") or 0) or random.randint(1, 2**31 - 1)
    wf = _sdxl_workflow(prompt, payload.get("negative_prompt") or "", w, h, steps, cfg, seed)

    # Start ComfyUI on demand (it idles at ~6 GB RAM; hearth stops it after a quiet
    # period — see comfy_supervisor). First image after idle pays a cold start.
    if not await comfy_supervisor.ensure_up():
        return _err(503, f"ComfyUI is not running at {_COMFY_URL} and hearth couldn't "
                         f"start it (set HEARTH_COMFY_DIR to your ComfyUI checkout)",
                    "backend_unavailable")

    img_bytes = None
    comfy_err = None
    comfy_supervisor.begin()  # suppress idle-stop while generating
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                r = await client.post(f"{_COMFY_URL}/prompt", json={"prompt": wf})
                if r.status_code != 200:
                    return _err(502, f"comfyui /prompt {r.status_code}: {r.text[:200]}", "backend_error")
                pid = r.json().get("prompt_id")
                if not pid:
                    return _err(502, "comfyui returned no prompt_id", "backend_error")
                # ComfyUI accepted the job → now free the chat model's VRAM for SDXL.
                # (Deferred to here so a failed/unreachable ComfyUI doesn't needlessly
                # evict the chat model.)
                await _free_ollama_vram(client)
                for _ in range(600):  # poll up to ~300s
                    await asyncio.sleep(0.5)
                    hist = (await client.get(f"{_COMFY_URL}/history/{pid}")).json().get(pid)
                    if not hist:
                        continue
                    images = [im for node in hist.get("outputs", {}).values()
                              for im in node.get("images", [])]
                    if images:
                        im = images[0]
                        v = await client.get(f"{_COMFY_URL}/view", params={
                            "filename": im.get("filename", ""), "subfolder": im.get("subfolder", ""),
                            "type": im.get("type", "output")})
                        if v.status_code == 200 and v.content:
                            img_bytes = v.content
                            break
                        # /view not ready/failed — keep polling (bounded by the loop)
                        continue
                    status = hist.get("status", {})
                    if status.get("status_str") == "error":  # fail fast, don't poll 300s
                        comfy_err = "; ".join(str(m) for m in status.get("messages", []))[:200] \
                                    or "ComfyUI execution error"
                        break
                    if status.get("completed"):
                        break
            finally:
                # Always release ComfyUI's VRAM (even on a mid-poll error) so the
                # chat model can reload (shared 12 GB GPU).
                try:
                    await client.post(f"{_COMFY_URL}/free",
                                      json={"unload_models": True, "free_memory": True}, timeout=10.0)
                except Exception:
                    pass
    except httpx.HTTPError as e:
        return _err(503, f"comfyui unreachable at {_COMFY_URL}: {e}", "backend_unavailable")
    finally:
        comfy_supervisor.end()  # generation done — let the idle timer run again
    if not img_bytes:
        return _err(502, f"image generation failed: {comfy_err or 'no output (timeout)'}", "backend_error")
    # Reload the chat model into the freed VRAM *after* the response is sent, so the
    # next message isn't a ~15-20s cold start. Use a BackgroundTask, not a bare
    # asyncio.create_task(): the event loop holds only a weak reference to a
    # fire-and-forget task, so it can be GC'd mid-await before it finishes (which is
    # exactly why the first attempt never reloaded). The response retains this one.
    return JSONResponse(
        {"created": int(time.time()),
         "data": [{"b64_json": base64.b64encode(img_bytes).decode()}]},
        background=BackgroundTask(_warm_chat, _cfg(request)))
