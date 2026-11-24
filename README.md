# mantel

**A universal, MCP-native desktop chat client — local-first, any backend.**

Mantel is a clickable chat app that talks to **any** model backend and defaults
to a **local [hearth](https://github.com/yumima/hearth) engine** on loopback. It
(will) host **MCP servers** so the model can use *your* tools — files, search,
databases, your own services — with **you** deciding which tools, regardless of
which model you run. Your data and tools stay local unless you explicitly opt
into a cloud backend or a remote server.

> **Status: M2** — a working desktop chat client that **hosts MCP servers**: the
> model can use *your* tools (deny-by-default), across any OpenAI-compatible
> backend, with a hearth-style management CLI. Multi-vendor adapters, profiles,
> and interactive approval are on the roadmap below.

## Why (vs. ChatGPT / Claude desktop)
- **No vendor lock-in.** One UI, any model — a free local model for most work, a
  frontier cloud model when you need it. The big clients are single-vendor.
- **Your tools, via MCP.** The tools the model can use are *yours* to configure —
  not a vendor's curated connector list — and work with whatever model you pick.
- **Local-first & private.** Defaults are local, no keys, no egress; cloud and
  remote MCP servers are explicit, labeled opt-ins.
- **Profiles** *(roadmap)* — one click swaps the whole stack: backend + model +
  MCP servers + system prompt.

## Quick start
```bash
git clone https://github.com/yumima/mantel && cd mantel
python3 -m venv .venv && .venv/bin/pip install -e ".[gui]"   # [gui] = native window (pywebview)

# with a hearth engine running on :11435 (see github.com/yumima/hearth):
.venv/bin/mantel            # opens the desktop chat window
# or:
.venv/bin/mantel serve      # headless — open http://127.0.0.1:8765 in a browser
.venv/bin/mantel install    # add a clickable launcher to your app menu
```
Without `[gui]`/pywebview, `mantel` falls back to a Chromium/Edge `--app` window,
then your default browser.

## Commands
| Command | What it does |
|---|---|
| `mantel` | open the desktop chat window (starts mantel's local server) |
| `mantel setup` | first-run wizard — find a backend, pick a model, seed MCP examples |
| `mantel status` | backend reachability, model, MCP servers + tool counts |
| `mantel doctor` | diagnose the environment (backend, MCP runtimes, window) |
| `mantel models` | list models from the active backend |
| `mantel config [show\|path\|edit]` | show / locate / edit the config file |
| `mantel mcp list\|tools\|enable <n>\|disable <n>` | manage MCP servers |
| `mantel provider list\|use <n>` | manage backends |
| `mantel serve` | run the local server in the foreground (headless / dev) |
| `mantel stop` / `restart` | stop / restart the local server |
| `mantel logs [-f]` | show (or follow) the local server log |
| `mantel install` / `uninstall` | add / remove the clickable desktop launcher |

## Architecture
Mantel runs a tiny **local server** (FastAPI) that serves the chat UI and proxies
chat/models to the configured backend under `/api/*`. The UI is **same-origin**
with mantel, so streaming `fetch` works with no CORS; mantel talks to the backend
**server-to-server** (so a loopback-only engine like hearth, with no CORS, works
fine). That proxy is exactly where the **MCP host + tool-call loop** land (M2).

```
window (pywebview)  →  mantel server (FastAPI)  →  backend (hearth / OpenAI / …)
   UI at  /               /api/chat  /api/models      /v1/chat/completions
                          + MCP host + agent loop ───→ MCP servers (your tools)
```

## Tools (MCP)
Mantel is an **MCP host**: the enabled servers' tools are offered to the model,
and mantel runs the **tool-calling loop** for you — executing calls, feeding
results back, looping until the model answers. Tool activity streams live into
the chat.

Execution is **deny-by-default**: a tool runs only if its name matches one of
its server's `auto_approve` globs — anything else is refused, so an autonomous
model can't touch files / shells / networks you didn't allow.

```bash
mantel setup                  # seeds safe, *disabled* example servers (needs npx)
mantel mcp list               # see configured servers
mantel mcp enable filesystem  # turn one on
mantel mcp tools              # what tools are now available
mantel                        # (re)open the window to pick up the change
```

Add your own with `mantel config edit`:
```yaml
mcp_servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "~/projects"]
    enabled: true
    auto_approve: ["read_*", "list_*"]   # reads auto-run; writes are refused
```
Any standard MCP **stdio** server works — `npx @modelcontextprotocol/server-*`,
`uvx mcp-server-*`, or one of your own.

**Manage it from the chat window** — no terminal needed. Type **`/`** for a
command palette that filters as you type; pick a command and it completes the
**argument too** — `/enable ` lists your servers (with on/off state), `/provider `
your backends, `/model ` the available models — so there's nothing to memorize.
↑/↓ to move, Enter/Tab to run, Esc to dismiss. Commands: `/tools`, `/servers`,
`/enable` / `/disable` (applied **live**, no restart), `/provider`, `/model`,
`/status`, `/clear`, `/help`.

## Voice
Press the waveform button (shown when the composer is empty) to talk. Mantel
answers when you pause, speaks sentence by sentence as the reply streams, and
stops the moment you start talking or tap the orb. **Hold to talk** (Space, or
press-and-hold the orb) suits noisy rooms or long pauses; **M** mutes, **Esc**
ends. The pause length and voice interruption are in Settings → Voice.

Under the hood: one mic stream feeds an AudioWorklet, so speech that cuts in
keeps its first syllables (pre-roll), and it is sent to whisper as 16 kHz WAV.
Replies are synthesized ahead of playback and scheduled gaplessly on WebAudio;
hearth keeps a warm Piper process per voice (~50 ms per sentence instead of
~350 ms).

## Learn (language tutor)
**Learn** in the sidebar opens a language home: level, streak, goal, and what's
due. Lessons are spoken (or typed), and the tutor remembers you between them:

- **Guided lesson**: warm-up recall of due items, one new skill, practice,
  recap. **Free conversation** and **Role-play** (café, doctor, job
  interview, … or your own scenario) are the other modes.
- Target-language phrases are highlighted and read by a native voice, with
  English in an English voice. Click any phrase to hear it again, which also
  saves it to your words.
- Ending a lesson extracts new words, corrections and a summary into a
  **spaced-repetition** store (Leitner boxes). Each mistake becomes a review
  card, and repeating it makes the card due again sooner.
- **Review** quizzes you out loud: say the answer, and whisper checks it.
  **Pronunciation** compares what you said to the target phrase word by word.

## Config
`~/.config/mantel/config.yaml` (optional — absent ⇒ local hearth defaults):
```yaml
host: 127.0.0.1
port: 8765
provider: hearth
model: primary_chat
web_search: true      # let hearth search the web in chats; false keeps them off it
providers:
  hearth: { type: openai, base_url: http://127.0.0.1:11435/v1 }
```

## Security
mantel binds **loopback only** and runs a **localhost guard** — a Host-header
allowlist plus cross-origin refusal — so a malicious web page in your browser
can't reach the server (DNS-rebinding / CSRF). That matters because the server
executes MCP tools and mutates config. Tool execution is **deny-by-default**;
note that `auto_approve` matches the **tool name the server reports**, so prefer
specific names over broad globs and use `*` only for servers you fully trust.
The config file (which may hold provider API keys) is written `0600`.

## Roadmap
- **M0 — scaffold + chat-to-hearth** ✅
- **M2 — MCP host + agent loop** ✅ — stdio servers, tool aggregation, the tool-calling loop, deny-by-default execution, live tool activity, a hearth-style management CLI (`setup` / `status` / `doctor` / `mcp` / `provider` / …).
- **M1 — multi-vendor backends** — Anthropic / Gemini adapters + key storage (any *OpenAI-compatible* backend already works via config).
- **M3 — profiles + settings UI** — profile switcher (backend + model + servers + prompt), server catalog, **interactive** tool approval.
- **M4 — connectors + packs** — credentialed servers (search / github / …), remote (SSE/HTTP) servers, domain packs.
- **M5 — distribution** — standalone bundles + GitHub releases.

## Develop
```bash
pip install -e ".[dev]"
pytest -q
```

## License
[Apache-2.0](LICENSE).
