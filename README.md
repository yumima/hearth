# hearth

**AI models on your own computer.** hearth picks a model that suits the machine,
checks that it works there, and serves it to your programs on one local address —
the API they already speak, OpenAI's. **mantel** is its chat window.

This repository is both: `packages/hearth` (the engine) and `packages/mantel` (the
[chat client](packages/mantel/README.md)). Two programs, installed together, that
speak over HTTP. [finterm](https://github.com/yumima/finterm) is one of hearth's
users; hearth knows nothing of finance.

## Today's commits (2026-11-24)

Latest first.

- [`a022eb0`](https://github.com/yumima/hearth/commit/a022eb0) Installer, client: what the review found
- [`7938097`](https://github.com/yumima/hearth/commit/7938097) Windows: programs that are running do not stop mantel from being added
- [`1531fc1`](https://github.com/yumima/hearth/commit/1531fc1) CI: the release's real Ollama and a small model, from download to removal
- [`9ef6dbe`](https://github.com/yumima/hearth/commit/9ef6dbe) Installer: asked to end in the middle, it goes back; a closed window stops nothing
- [`ccca8d7`](https://github.com/yumima/hearth/commit/ccca8d7) hearth client install; the installer recognises what is its own
- [`d5778de`](https://github.com/yumima/hearth/commit/d5778de) catalog: the 9B, measured
- [`5f4a48a`](https://github.com/yumima/hearth/commit/5f4a48a) Installer: what the review found
- [`0d99546`](https://github.com/yumima/hearth/commit/0d99546) README, release and site workflows (a tag makes a draft), the site
- [`5095592`](https://github.com/yumima/hearth/commit/5095592) Windows: stopping a stalled download; what is installed, and its licences
- [`684cff7`](https://github.com/yumima/hearth/commit/684cff7) catalog: what was measured here; two models that failed leave it
- [`1252e78`](https://github.com/yumima/hearth/commit/1252e78) Install, update, go back, remove — tested with the scripts themselves
- [`23e1c91`](https://github.com/yumima/hearth/commit/23e1c91) Windows: a port in use is not shared
- [`c98309d`](https://github.com/yumima/hearth/commit/c98309d) Models: what the review found; the installer's first parts
- [`f22d302`](https://github.com/yumima/hearth/commit/f22d302) Models that suit this computer: found, ranked, checked here, then used
- [`498a50c`](https://github.com/yumima/hearth/commit/498a50c) Ollama's owner, the stop, and the client: what the review found
- [`0b6e65f`](https://github.com/yumima/hearth/commit/0b6e65f) stop: only a gateway's own process, not one that mentions hearth
- [`07fad80`](https://github.com/yumima/hearth/commit/07fad80) One lock file, two environments; CI on Linux, macOS and Windows
- [`0b702ff`](https://github.com/yumima/hearth/commit/0b702ff) Ollama kept running, one way to handle processes everywhere; the chat window is mantel
- [`fccccf8`](https://github.com/yumima/hearth/commit/fccccf8) layout: hearth under packages/
- [`279c827`](https://github.com/yumima/hearth/commit/279c827) start --background: what the review found
- [`6d22b0c`](https://github.com/yumima/hearth/commit/6d22b0c) **ci:** the engine branch too
- [`876842f`](https://github.com/yumima/hearth/commit/876842f) **ci:** the engine branch too
- [`1edae77`](https://github.com/yumima/hearth/commit/1edae77) Voice and web search: what the review found
- [`83f1dc3`](https://github.com/yumima/hearth/commit/83f1dc3) start --background: the gateway outlives whoever started it
- [`0e41e46`](https://github.com/yumima/hearth/commit/0e41e46) Fresh installs work again (mcp 1.x); ask hearth for web search
- [`efbb107`](https://github.com/yumima/hearth/commit/efbb107) 0.2.0: web search only for a request that asks; other web pages are refused
- [`0db91e9`](https://github.com/yumima/hearth/commit/0db91e9) voice: choose the English voice; speech goes to a voice that can pronounce it

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
do, and lists the models that suit it. Options: `--no-mantel`, `--no-autostart`,
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
| `hearth plan` | which models suit this computer |
| `hearth setup` | install the recommended one, and natural voices |
| `hearth install <model>` · `remove <model>` | what the setup page does, here |
| `hearth start` · `start --background` · `stop` · `status` | the gateway (the service, where one is installed) |
| `hearth` · `hearth chat` | chat in the terminal (`/help`) |
| `hearth client` | mantel's window, with the engine started |
| `hearth client install` | build mantel if it is not here (from this source tree, or the release), and add it to the app menu; `--build` to build it again, `--replace` to take over `~/.local/bin/mantel` |
| `hearth client uninstall` | take it out of the app menu |
| `hearth code` | a coding agent for a folder |
| `hearth bind <role> <model>` · `roles` · `models` | which model answers for what |
| `hearth backend …` | add a cloud or other OpenAI-compatible server |
| `hearth voice --voice kokoro` | natural voices |
| `hearth service autostart on\|off` | start at login |
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
run); rebind with `hearth bind <role> <model> --context <tokens>` (persisted +
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
make build     # hearth in .venv, mantel in .venv-mantel
make test      # both, as CI runs them
hearth client install    # mantel from this tree, in the app menu
make start     # the gateway; Ollama from vendor/ollama, or on PATH
HEARTH_E2E=1 .venv/bin/python -m pytest tests/test_install.py   # the install scripts, with real environments
python3 tools/release.py OUT                                    # a release's files
```

Dependencies are locked with hashes (`uv.lock`); `make lock` after changing one.
An installed hearth and a checkout do not replace each other: the installer steps
aside for a hearth it did not make.

The consumer contract (what finterm and mantel rely on) is versioned:
`GET /admin/version` says `contract`. Its major number rises when a consumer has
to change; additions raise the minor one.

License: Apache-2.0.
