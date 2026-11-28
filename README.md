# hearth

**Local AI engine.** Open-weight models (e.g. Qwen) on Ollama, fitted to local
hardware (GPU, VRAM, RAM), with an OpenAI-compatible API for chat, coding, voice,
agents and MCP tools. Free to use, no signup. Linux, Windows, macOS. `hearth` in a
folder starts the coding agent; **Hearth** in the app menu, the chat window.

This repository is all of it: `packages/hearth` (the engine and the command) and
`packages/client` (the [chat window](packages/client/README.md)). Two programs,
installed together, that speak over HTTP. [finterm](https://github.com/yumima/finterm) is one of hearth's
users; hearth knows nothing of finance.

## Today's commits (2026-11-27)

Latest first.

- [`d7ba3a5`](https://github.com/yumima/hearth/commit/d7ba3a5) release: build SearXNG from a git archive, not a checkout
- [`51012d0`](https://github.com/yumima/hearth/commit/51012d0) release: check out only what SearXNG's wheel is built from
- [`0720a91`](https://github.com/yumima/hearth/commit/0720a91) hearth 0.2.1: SearXNG is the search backend, installed with every release
- [`af073d0`](https://github.com/yumima/hearth/commit/af073d0) installer: offer a SearXNG of hearth's own for web search
- [`337f3af`](https://github.com/yumima/hearth/commit/337f3af) websearch: report HTTP failures as 'HTTP 429 …' so the whole chain fits the card

[See all commits →](https://github.com/yumima/hearth/commits/main)

## Install

Linux (x86-64) and macOS (Apple Silicon):

```bash
curl -fsSL https://yumima.github.io/hearth/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://yumima.github.io/hearth/install.ps1 | iex
```

For one user, without administrator rights, into one folder. An Ollama, Python or
hearth that is already there is not touched. The script fetches exactly the files
whose hashes it carries. Then the setup page opens: it says what the computer can
do, and lists the models that suit it. Options: `--no-client`, `--no-autostart`,
`--no-open`.

| | |
|---|---|
| `hearth update` | to the newest release; one that does not start is taken back by itself |
| `hearth update --rollback` | to the version before, with the settings and chats of then |
| `hearth uninstall` | removes it (`--keep-models`, `--keep-settings`) |

**What it runs on.** An NVIDIA or AMD card, or Apple Silicon; without one, a small
model on the processor, slowly. Run by us on NVIDIA under Linux; the others are
**experimental** until somebody has told us how they do. See
[what is installed, and its licences](THIRD_PARTY.md).

## The setup page

<http://127.0.0.1:11435/> — which models suit this computer, in order; install, use,
remove; natural voices and speech recognition. A model is downloaded, compared with
the build that was tested, loaded with as much context as stays on the GPU,
measured, and checked (JSON, tools, figures, long texts) before any program gets
it. What was in use stays in use if that fails.

## Commands

| Command | What it does |
|---|---|
| `hearth` | a coding agent for the folder you are in: it reads, and asks before it changes a file, runs a command or goes to the web. It asks once whether you trust the folder, and starts the engine if need be. `hearth code FOLDER` for another |
| `hearth -p "…"` | one task, without asking, and the answer; for scripts |
| `hearth chat` | chat in the terminal, without files (`/chat` and `/code` go from one to the other) |
| `hearth client` | the chat window, with the engine started. `client install` builds it and adds **Hearth** to the app menu, `client uninstall` takes it out; `client --help` has its own commands |
| `hearth setup` | install the model that suits this computer, and natural voices |
| `hearth model list` · `suggest` | what is here and what it is used for · what suits this computer |
| `hearth model install M` · `use M` · `remove M` | download, check and use a model · use another (`--for ROLE`, `--provider P`) · delete its files |
| `hearth voice list` · `install V` · `remove V` | voices for reading aloud: `kokoro` (the natural ones) or a Piper voice |
| `hearth provider list` · `add` · `remove` | a cloud, or another server that speaks OpenAI's API |
| `hearth start` · `stop` · `restart` · `status` · `logs` | the engine (the service, where one is installed) |
| `hearth autostart on\|off` | start it at login |
| `hearth serve` | the engine in the foreground: what a service runs |
| `hearth search "…"` | the engine's web search, without a model |
| `hearth update` · `uninstall` | see above |

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
| `GET /admin/setup` · `POST /admin/jobs` · `POST /admin/jobs/{id}/stop` | the setup page's: the plan, what is installed; install a model or voices |
| `DELETE /admin/models/{id}` | remove a model no role uses |
| `POST /admin/shutdown` | end the gateway in order |

### Role registry

`model: "primary_chat"` → role registry → concrete `(model_id, backend)`. A
literal id (`qwen3.8:27b`) routes straight through, using the context of any
role bound to it. Config at `~/.hearth/config.yaml` (written on first
run); choose another with `hearth model use <model> --for <role> --context <tokens>` (persisted +
hot-applied). Omitting `--context` matches other roles on that model, or else
clears the role's context.

`hearth setup` and the setup page bind the chat roles to one model, so a hop from
one role to another does not swap models, and `embedding` to `nomic-embed-text` —
which they never change again: what other programs have embedded depends on it.

**Model fit.** The governing rule is to fit the model entirely in the GPU's memory.
One that spills runs its overflow layers on the processor and leaves the GPU
waiting: the full 27B managed 3 tokens a second on a 12 GB card, its 2-bit build
27. What is offered, with measured sizes, is in
[`catalog.json`](packages/hearth/src/hearth/catalog.json).

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

With [uv](https://docs.astral.sh/uv):

```bash
make build     # the engine in .venv, the window in .venv-client
make test      # both, as CI runs them
hearth client install    # the window from this tree, in the app menu
make start     # the gateway; Ollama from vendor/ollama, or on PATH
HEARTH_E2E=1 .venv/bin/python -m pytest tests/test_install.py   # the install scripts, with real environments
python3 tools/release.py OUT                                    # a release's files
```

Dependencies are locked with hashes (`uv.lock`); `make lock` after changing one.
An installed hearth and a checkout do not replace each other: the installer steps
aside for a hearth it did not make.

The consumer contract (what finterm and the window rely on) is versioned:
`GET /admin/version` says `contract`. Its major number rises when a consumer has
to change; additions raise the minor one.

License: Apache-2.0.
