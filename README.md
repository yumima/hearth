# hearth

A stable, **OpenAI-compatible local AI engine** on loopback. Hearth fronts
[Ollama](https://ollama.com) (and, later, llama.cpp / vLLM / faster-whisper)
behind one HTTP API with a **role registry**, **hardware probe**, and (M2)
**tool-call repair**.

**finterm** is its first consumer — but the design has zero finance knowledge.
Any `openai`-SDK client works pointed at the base URL. The full design spec
lives in finterm's `plans/local-ai-engine.md`.

## Today's commits (2026-06-08)

Latest first.


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
hearth install        # …or install the clickable desktop chat app (app-menu icon)
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
| `hearth` / `hearth chat` | streaming chat — **bare `hearth` launches it** (like `claude`). Live thinking spinner + deep-think toggle. `/help` lists in-chat commands: `/model` (picker), `/think on\|off`, `/system`, `/status`, `/retry`, `/save`, `/clear`, … Flags `--model` `--system` `--no-think` `--show-thinking` |
| `hearth pull <model>` | pull a model via Ollama |
| `hearth bind <role> <model>` | rebind a role (persisted + hot-applied to a running gateway) |
| `hearth roles` / `models` | show role bindings / servable models |
| `hearth hardware` | GPU / VRAM / RAM probe |
| `hearth install` / `uninstall` | install / remove the **desktop chat app** — a clickable launcher (app menu / Launchpad / Start menu) for the GUI client, cross-platform via the OS's native webview |
| `hearth gui` | open the desktop chat window now (this is what the app icon runs) |
| `hearth service …` | control a background systemd --user gateway service: `start` `stop` `restart` `status` `logs` `autostart on\|off` |

Equivalent `make` targets exist (`make start|stop|status|chat|test`).

### Desktop chat app

The clickable counterpart to the terminal client — a standalone chat window,
like the ChatGPT / Claude desktop apps. The gateway serves the UI at `/app`
(same-origin, so streaming + the localhost-only security model both just work),
and the window is a chrome-less native frame via each OS's own webview
(WebView2 / WKWebView / WebKitGTK — no bundled browser).

```bash
hearth install     # add it to your menu — Linux .desktop / macOS .app / Windows .lnk
hearth gui         # open the window now (starts the engine if it isn't running)
hearth uninstall   # remove the launcher
# or: make install-client / make uninstall-client
```

Clicking the launcher starts the engine on demand if needed, then opens the
window. For the most native window, `pip install 'hearth[gui]'` (pywebview);
otherwise it falls back to a Chromium/Edge `--app` window, then your browser.
The UI is also reachable at <http://localhost:11435/app> in any browser.

### Always-on engine (optional)

`hearth gui` (the desktop app) and `hearth start` launch the engine on demand,
which covers normal use. To keep it **always** up in the background — e.g. for
finterm or an OpenAI-SDK client when no window is open — run `hearth start` from
your login autostart, or a **systemd --user** unit that runs it. `hearth
service` drives such a unit once it exists:

```bash
hearth service start | stop | restart | status
hearth service logs -f               # follow logs (journal)
hearth service autostart on --boot   # start on login / at boot (linger)
```

(hearth no longer installs the unit itself — the desktop app + on-demand start
replaced `service install`; create a unit with your init system if you want it
always-on.)

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

| Role | Built-in default | `setup` picks (largest that fits VRAM) |
|---|---|---|
| `primary_chat` | `qwen3:14b` | 12 GB → `qwen3:14b` · 24 GB → `qwen3:30b-a3b` · 32 GB → `qwen3:32b` |
| `fast_chat` / `coding` | `qwen3:8b` | `qwen3:8b` |
| `embedding` | `nomic-embed-text` | `nomic-embed-text` |
| `stt` | `faster-whisper:medium` *(M2)* | — |

**Model fit (`hearth setup`).** The wizard probes GPU VRAM + system RAM and picks
the **largest Qwen3 model that fits entirely in VRAM** — that's the rule that
matters. A model that overflows VRAM (e.g. the 30B on a 12 GB GPU) runs its
spillover layers on the CPU, leaving the GPU mostly idle waiting on them
(measured ~12% GPU utilisation, far slower than a smaller model sitting 100%
on-GPU); the MoE's few-active-params doesn't rescue this. So a 12 GB GPU gets
`qwen3:14b` (fits → ~45 tok/s at 99% GPU), a 24 GB GPU gets `qwen3:30b-a3b`,
32 GB+ gets `qwen3:32b` (thresholds keep ~2–3 GB headroom for the KV cache);
CPU-only boxes are sized by RAM. Precedence: built-in default
→ `~/.hearth/config.yaml` (`setup`/`bind`) → per-request `model`.

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
