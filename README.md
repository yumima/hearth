# hearth

A stable, **OpenAI-compatible local AI engine** on loopback. Hearth fronts
[Ollama](https://ollama.com) (and, later, llama.cpp / vLLM / faster-whisper)
behind one HTTP API with a **role registry**, **hardware probe**, and (M2)
**tool-call repair**.

**finterm** is its first consumer — but the design has zero finance knowledge.
Any `openai`-SDK client works pointed at the base URL. The full design spec
lives in finterm's `plans/local-ai-engine.md`.

## Why

Consumers integrate once against the OpenAI contract; the engine swaps model
backends underneath. You don't hand-manage Ollama, pick concrete model names,
or write backend-specific code — you ask for a *role* (`primary_chat`,
`fast_chat`, `embedding`, …) and hearth resolves it.

## Quick start

```bash
cd hearth
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

# One command brings up the bundled Ollama + the gateway:
hearth start
#   -> serves http://127.0.0.1:11435  (one off from Ollama's 11434)

# In another shell — auto-detect your hardware, pull a model that fits, bind roles:
hearth setup          # probe → recommend (e.g. qwen3:30b-a3b on a 12GB GPU + ample RAM) → pull → bind

# …or pull specific models yourself:
hearth pull qwen3:14b          # primary_chat (fully on a 12GB GPU)
hearth pull nomic-embed-text   # embedding

hearth roles          # show role -> model bindings
hearth hardware       # GPU / VRAM / RAM probe
```

The bundled Ollama binary lives under `vendor/` (downloaded, not committed). It
ships its own CUDA runtime, so on an NVIDIA box it uses the GPU with no system
CUDA toolkit.

## Manage it

```bash
hearth status         # is the gateway up? show role bindings
hearth stop           # stop a gateway started with `hearth start`
hearth chat           # interactive streaming chat in the terminal (ChatGPT-style)
hearth chat --model fast_chat --system "Be terse."
```

Or via `make` (build / start / stop / status / chat / test). And for a
clickable launcher in your app menu (Linux / freedesktop):

```bash
make build            # venv + editable install
make install-app      # adds a "hearth" entry to your application menu
make uninstall-app    # removes it
```

`make install-app` writes `~/.local/share/applications/hearth.desktop`;
clicking it starts the engine detached (logs to `~/.hearth/hearth.log`) and
shows a desktop notification. Stop it with `hearth stop`.

## API

OpenAI-compatible (point any `openai` client here):

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming SSE + tool calling |
| `POST /v1/embeddings` | text → vector |
| `GET /v1/models` | concrete models **+** role aliases |

Admin (off `/v1` so OpenAI clients never see it):

| Endpoint | Notes |
|---|---|
| `GET /admin/health` `GET /admin/ready` | liveness / readiness |
| `GET /admin/hardware` | GPU type, VRAM, RAM, best inference target |
| `GET /admin/backends` | installed backends + capability flags |
| `GET /admin/roles` · `PUT /admin/roles/{role}` | view / hot-rebind a role |
| `POST /admin/models/pull` · `DELETE /admin/models/{id}` | lifecycle |

### Role registry

`model: "primary_chat"` → role registry → concrete `(model_id, backend)`. A
literal id (`qwen2.5:14b`) routes straight through. Config at
`~/.hearth/config.yaml` (written on first run); rebind with
`hearth bind primary_chat qwen2.5:7b-instruct-q4_K_M`.

| Role | Default (12 GB-VRAM box) |
|---|---|
| `primary_chat` | `qwen2.5:14b-instruct-q4_K_M` |
| `fast_chat` / `coding` | `qwen2.5:7b-instruct-q4_K_M` |
| `embedding` | `nomic-embed-text` |
| `stt` | `faster-whisper:medium` *(M2)* |

## Security

Single-user, single-host, **loopback-bound by default**. Host-header guard +
no CORS mitigates DNS-rebinding. Off-loopback needs `--unsafe-bind`, which
forces `--api-key`.

## Develop

```bash
pip install -e ".[dev]"
pytest                       # unit tests (no Ollama needed)
python tests/smoke_live.py   # live smoke (needs the stack up + models pulled)
```

## Status

M1 — walking skeleton (this): gateway + Ollama adapter + role registry + CLI +
hardware probe. Roadmap (STT, tool-call repair, concurrency, lifecycle wizard,
second backend) in the design doc §5.

License: Apache-2.0.
