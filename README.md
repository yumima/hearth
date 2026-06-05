# hearth

A stable, **OpenAI-compatible local AI engine** on loopback. Hearth fronts
[Ollama](https://ollama.com) (and, later, llama.cpp / vLLM / faster-whisper)
behind one HTTP API with a **role registry**, **hardware probe**, and (M2)
**tool-call repair**.

**finterm** is its first consumer — but the design has zero finance knowledge.
Any `openai`-SDK client works pointed at the base URL. The full design spec
lives in finterm's `plans/local-ai-engine.md`.

## Today's commits (2026-06-05)

Latest first.

- [`2193608`](https://github.com/yumima/hearth/commit/2193608) chat: deep-think on/off toggle + live thinking spinner
- [`39b28ac`](https://github.com/yumima/hearth/commit/39b28ac) cli: show reasoning-model thinking in chat; stop-by-port fallback
- [`9222a87`](https://github.com/yumima/hearth/commit/9222a87) **docs:** refresh README + auto-maintained 'Today's commits' changelog
- [`f0106e8`](https://github.com/yumima/hearth/commit/f0106e8) make: add install-cli (symlink hearth onto ~/.local/bin PATH)
- [`589d8b1`](https://github.com/yumima/hearth/commit/589d8b1) setup: first-run hardware-fit wizard + Qwen3 defaults
- [`9f6428e`](https://github.com/yumima/hearth/commit/9f6428e) cli: harden chat stream parse + pidfile lifecycle; quote launcher Exec
- [`bf2b76d`](https://github.com/yumima/hearth/commit/bf2b76d) cli: add chat/status/stop, Makefile, clickable desktop app

[See all commits →](https://github.com/yumima/hearth/commits/main)

## Why

Consumers integrate once against the OpenAI contract; the engine swaps model
backends underneath. You don't hand-manage Ollama, pick concrete model names,
or write backend-specific code — you ask for a *role* (`primary_chat`,
`fast_chat`, `embedding`, …) and hearth resolves it.

## Quick start

```bash
cd hearth
make build            # python venv + editable install
make install-cli      # symlink `hearth` into ~/.local/bin so it's on PATH
                      # (open a new shell afterwards, or `hash -r`)

hearth start          # bring up the bundled Ollama + the gateway on :11435
                      #   (leave running; Ctrl-C stops it)

# …then in a second shell:
hearth setup          # probe hardware → pull a model that fits → bind roles
hearth chat           # interactive chat in the terminal (ChatGPT-style)
```

`make build` is `python3 -m venv .venv && .venv/bin/pip install -e .` — you can
also run `.venv/bin/hearth …` directly without `install-cli`.

The bundled Ollama binary lives under `vendor/` (downloaded, not committed). It
ships its own CUDA runtime, so on an NVIDIA box it uses the GPU with no system
CUDA toolkit.

## Commands

| Command | What it does |
|---|---|
| `hearth start` | bring up Ollama + serve the gateway (foreground) |
| `hearth stop` | stop the running gateway (found by pidfile, else by its port) |
| `hearth status` | is the gateway up? show role bindings |
| `hearth setup` | **first-run wizard** — probe hardware, pull a fitting model, bind roles |
| `hearth chat` | streaming chat with a live thinking spinner + deep-think toggle (`/think on\|off`, default on; off = fast lane). Flags `--model` `--system` `--no-think` `--show-thinking`; in-chat `/exit` `/reset` `/model` `/show` |
| `hearth pull <model>` | pull a model via Ollama |
| `hearth bind <role> <model>` | rebind a role (persisted + hot-applied to a running gateway) |
| `hearth roles` / `models` | show role bindings / servable models |
| `hearth hardware` | GPU / VRAM / RAM probe |
| `hearth service …` | run as a background systemd --user service: `install` `start` `stop` `restart` `status` `logs` `autostart on\|off` |

Equivalent `make` targets exist (`make start|stop|status|chat|test`). For a
**clickable launcher** in your app menu (Linux / freedesktop):

```bash
make install-app      # writes ~/.local/share/applications/hearth.desktop
make uninstall-app
```

Clicking it starts the engine detached (logs to `~/.hearth/hearth.log`) and
shows a desktop notification. Stop it with `hearth stop`.

### Run as a background service

For an always-available engine (CLI chat, finterm, any OpenAI-SDK client), run
hearth as a **systemd --user service** — driven entirely through `hearth`, no
`systemctl` needed:

```bash
hearth service install --autostart   # install the unit + start on login
hearth service start                 # start it now (frees your terminal)
hearth service status                # is it up?
hearth service logs -f               # follow logs (journal)
hearth service autostart on --boot   # also start at boot, before login (linger)
hearth service stop | restart | uninstall
```

The unit runs `hearth start` (gateway **+** Ollama, cleanly stopped together via
the service cgroup) and is pinned to the stable `~/.local/bin/hearth` symlink, so
rebuilding the venv doesn't break it. After editing hearth's code,
`hearth service restart` reloads it.

## API

OpenAI-compatible (point any `openai` client here):

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming SSE + tool calling; optional `think: bool` (hearth extension) toggles a reasoning model's chain-of-thought via the native API |
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
literal id (`qwen3:14b`) routes straight through. Config at
`~/.hearth/config.yaml` (written on first run); rebind with
`hearth bind primary_chat qwen3:8b` (persisted + hot-applied).

Built-in defaults (Qwen3); `hearth setup` overrides them per your hardware:

| Role | Built-in default | `setup` on a 12 GB GPU + ≥24 GB RAM |
|---|---|---|
| `primary_chat` | `qwen3:14b` | `qwen3:30b-a3b` (MoE — see below) |
| `fast_chat` / `coding` | `qwen3:8b` | `qwen3:8b` |
| `embedding` | `nomic-embed-text` | `nomic-embed-text` |
| `stt` | `faster-whisper:medium` *(M2)* | — |

**Model fit (`hearth setup`).** The wizard probes GPU VRAM + system RAM and picks
a Qwen3 model that fits. On a ~12 GB GPU with ample RAM it prefers the **30B MoE**
(`qwen3:30b-a3b`): only ~3B params are active per token, so the few GB that spill
from VRAM to RAM cost little speed while beating a dense 14B on quality. Smaller
GPUs get dense models that fit fully on-GPU; CPU-only boxes are sized by RAM.
Precedence: built-in default → `~/.hearth/config.yaml` (`setup`/`bind`) →
per-request `model`.

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

M1 (gateway + Ollama adapter + role registry + CLI + hardware probe) and the
first-run **hardware-fit `setup` wizard** are done. Roadmap (STT via
faster-whisper, tool-call repair, concurrency limits, a second backend) is in
the design doc §5.

License: Apache-2.0.
