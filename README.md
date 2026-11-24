# hearth

A stable, **OpenAI-compatible local AI engine** on loopback. Hearth fronts
[Ollama](https://ollama.com) (and, later, llama.cpp / vLLM / faster-whisper)
behind one HTTP API with a **role registry**, **hardware probe**, and (M2)
**tool-call repair**.

**finterm** is its first consumer — but the design has zero finance knowledge.
Any `openai`-SDK client works pointed at the base URL. The full design spec
lives in finterm's `plans/local-ai-engine.md`.

## Today's commits (2026-11-24)

Latest first.

- [`876842f`](https://github.com/yumima/hearth/commit/876842f) **ci:** the engine branch too
- [`83f1dc3`](https://github.com/yumima/hearth/commit/83f1dc3) start --background: the gateway outlives whoever started it
- [`efbb107`](https://github.com/yumima/hearth/commit/efbb107) 0.2.0: web search only for a request that asks; other web pages are refused

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
hearth client install # …or install the clickable desktop chat app (app-menu icon)
```

`make build` is `python3 -m venv .venv && .venv/bin/pip install -e .` — you can
also run `.venv/bin/hearth …` directly without `install-cli`.

The bundled Ollama binary lives under `vendor/` (downloaded, not committed). It
ships its own CUDA runtime, so on an NVIDIA box it uses the GPU with no system
CUDA toolkit. Keep it current: Qwen 3.8 is tested on Ollama 0.40.14.

## Commands

| Command | What it does |
|---|---|
| `hearth start` | bring up Ollama + serve the gateway (foreground) |
| `hearth start --background` | the same, returning once it answers; it keeps running (the service, where one is installed) |
| `hearth stop` | stop the running gateway (found by pidfile, else by its port) |
| `hearth status` | is the gateway up? show role bindings |
| `hearth setup` | **first-run wizard** — probe hardware, pull a fitting model, bind roles |
| `hearth` / `hearth chat` | streaming chat — **bare `hearth` launches it** (like `claude`). Live thinking spinner + deep-think toggle. `/help` lists in-chat commands: `/model` (picker), `/think on\|off`, `/system`, `/status`, `/retry`, `/save`, `/clear`, … Flags `--model` `--system` `--no-think` `--show-thinking` |
| `hearth pull <model>` | pull a model via Ollama |
| `hearth bind <role> <model>` | rebind a role (persisted + hot-applied to a running gateway) |
| `hearth roles` / `models` | show role bindings / servable models |
| `hearth hardware` | GPU / VRAM / RAM probe |
| `hearth client …` | **desktop chat app** — bare `hearth client` opens the window; `hearth client install` / `uninstall` manage the clickable launcher (app menu / Launchpad / Start menu), cross-platform via the OS's native webview |
| `hearth service …` | control a background systemd --user gateway service: `start` `stop` `restart` `status` `logs` `autostart on\|off` |

Equivalent `make` targets exist (`make start|stop|status|chat|test`).

### Desktop chat app

The clickable counterpart to the terminal client — a standalone chat window,
like the ChatGPT / Claude desktop apps. The gateway serves the UI at `/app`
(same-origin, so streaming + the localhost-only security model both just work),
and the window is a chrome-less native frame via each OS's own webview
(WebView2 / WKWebView / WebKitGTK — no bundled browser).

```bash
hearth client install     # add it to your menu — Linux .desktop / macOS .app / Windows .lnk
hearth client             # open the window now (starts the engine if it isn't running)
hearth client uninstall   # remove the launcher
# or: make install-client / make uninstall-client
```

Clicking the launcher starts the engine on demand if needed, then opens the
window. For the most native window, `pip install 'hearth[gui]'` (pywebview);
otherwise it falls back to a Chromium/Edge `--app` window, then your browser.
The UI is also reachable at <http://localhost:11435/app> in any browser.

### Always-on engine (optional)

`hearth client` (the desktop app) and `hearth start` launch the engine on demand,
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
| `POST /v1/chat/completions` | streaming SSE + tool calling; optional `think: bool` (hearth extension) toggles a reasoning model's chain-of-thought via the native API; `web_search: true` (also an extension) adds the engine's web tools |
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
literal id (`qwen3.8:27b`) routes straight through, using the context of any
role bound to it. Config at `~/.hearth/config.yaml` (written on first
run); rebind with `hearth bind <role> <model> --context <tokens>` (persisted +
hot-applied). Omitting `--context` matches other roles on that model, or else
clears the role's context.

Built-in defaults; `hearth setup` overrides them per your hardware:

| Role | Built-in default | `setup` picks (largest that fits VRAM) |
|---|---|---|
| `primary_chat` | 2-bit Qwen 3.8 27B | 20 GB+ → `qwen3.8:27b` · 11–20 GB → 2-bit Qwen 3.8 27B · 7–11 GB → `qwen3:8b` |
| `fast_chat` / `coding` | 2-bit Qwen 3.8 27B | 20 GB+ → `qwen3.8:27b` · 11–20 GB → 2-bit Qwen 3.8 27B · 7–11 GB → `qwen3:4b` |
| `embedding` | `nomic-embed-text` | `nomic-embed-text` |
| `stt` | `faster-whisper:medium` *(M2)* | — |

**Model fit (`hearth setup`).** The wizard probes GPU VRAM + system RAM and picks
the **largest model that fits entirely in VRAM**. A model that overflows runs its
spillover layers on the CPU, leaving the GPU mostly idle waiting on them. On a
12 GB GPU the full `qwen3.8:27b` managed only about 3 tok/s that way, so 11–20 GB
cards get `hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ2_S`, a 2-bit community build that
fits fully at a 40960 context and trades some accuracy for usable speed. 20 GB+
cards get the full model.
CPU-only boxes are sized by RAM. Precedence: built-in default
→ `~/.hearth/config.yaml` (`setup`/`bind`) → per-request `model`.

## Security

Single-user, single-host, **loopback-bound by default**. A Host-header guard and
no CORS mitigate DNS-rebinding, and a request sent by another web page is refused
(a browser names its origin; only hearth's own is accepted). Off-loopback needs
`--unsafe-bind`, which forces `--api-key`. On a computer shared by several
accounts, any of them can reach the loopback port.

**Web search is only for requests that ask.** The engine adds `web_search` and
`web_fetch` when the request says `"web_search": true`, never on its own: a chat
holding private data plus tools that reach the internet is a way for a hostile
page to get that data out. `hearth chat` and the chat window ask, unless
`search.chat: false` is in config.yaml; `hearth code` carries the same tools
itself; `search.enabled: false` turns all of it off.

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
