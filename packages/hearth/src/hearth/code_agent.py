"""hearth code — a local coding agent for a project directory.

A tool-calling agent bound to the ``coding`` role that explores and edits a
project through hearth's own gateway, looping until the task is done. It is a
lean, in-process alternative to Claude Code / aider for when you want the agent
to live in hearth itself and talk to a local model.

What makes an agent effective at this is mostly its tools, and three of them
matter more than the rest:

* **grep** — a model that cannot search reads whole files to find a symbol, and
  a few of those exhaust the context window before any work starts.
* **edit_file** — an exact-string replacement, not a whole-file rewrite. Asking
  a local model to re-emit a 600-line file to change two of them wastes output
  tokens, and it reliably drops formatting and unrelated code on the way.
* **read_file with line numbers and a window** — so a large file can be read in
  parts and referred to precisely.

Every path is confined to the project root, links included. Reading there is free
(the person was asked once whether they trust the folder, cli.py); a change, a
command and the web each ask first, and show what they would do: a file that was
read may hold instructions of somebody else's, and a web address is a way out.
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import config as cfgmod
from . import procs, toolloop, webtools
from .term import ask

_BOLD, _DIM, _YEL, _CYAN, _RST = "\033[1m", "\033[2m", "\033[33m", "\033[36m", "\033[0m"

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist",
              ".mypy_cache", ".pytest_cache", ".ruff_cache", ".next", "target"}
# Read caps. A single read must not be able to swallow the whole window: at
# ~4 chars/token a 60k-char file is ~15k tokens, which overruns most local
# context windows on its own and silently evicts the system prompt.
_READ_LIMIT_LINES = 1200
_READ_MAX_CHARS = 40000
_GREP_MAX = 120
_LIST_MAX = 600
_SHELL_MAX_CHARS = 16000
_FILE_MAX_BYTES = 20_000_000
_DIFF_LINES = 60
_DIFF_CHARS = 8000

_FILE_TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List project files, optionally under a subdirectory and/or matching a "
                       "glob pattern (e.g. '*.py', 'src/**/*.ts'). Use this to get oriented.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "subdirectory relative to the project root"},
            "pattern": {"type": "string", "description": "glob filter on the file name/path"}}}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file CONTENTS with a regular expression and return matching "
                       "lines as 'path:line: text'. This is the fastest way to locate a "
                       "symbol, string or definition — prefer it over reading files to look "
                       "for something.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression"},
            "path": {"type": "string", "description": "subdirectory to search (default: whole project)"},
            "glob": {"type": "string", "description": "only search files matching this glob, e.g. '*.py'"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file and return it with line numbers. Large files are "
                       "windowed — pass offset/limit to read further.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based first line (default 1)"},
            "limit": {"type": "integer", "description": f"lines to return (default {_READ_LIMIT_LINES})"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace an EXACT string in a file. This is the preferred way to change "
                       "code. old_string must match the file byte-for-byte (including "
                       "indentation) and must be unique — include surrounding lines for "
                       "context if it is not. Read the file first.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "exact text to replace"},
            "new_string": {"type": "string", "description": "replacement text"},
            "replace_all": {"type": "boolean", "description": "replace every occurrence"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create a new file, or overwrite one completely with its FULL new "
                       "content. Use edit_file instead when changing part of an existing file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command in the project root and return combined "
                       "stdout+stderr with the exit code. Use it to build, run tests, or "
                       "inspect the environment.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "seconds (default 120)"}},
            "required": ["command"]}}},
]


def _fmt_size(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def _plain(text: str) -> str:
    """As it is shown to somebody who decides on it: nothing in it moves the cursor,
    changes a colour or writes over what was there."""
    return "".join(c if c == "\n" or c == "\t" or (c.isprintable() and not "\u202a" <= c <= "\u202e"
                                                  and not "\u2066" <= c <= "\u2069") else f"\\x{ord(c):02x}" for c in text)


def _diff(old: str, new: str, name: str) -> str:
    """What would change. All of it is shown, or it is said that it is not."""
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), name, name, lineterm="", n=2))[2:]
    out, left = [], _DIFF_CHARS
    for n, line in enumerate(lines):
        text = _plain(line)
        if n >= _DIFF_LINES or len(text) > left:
            rest = len(lines) - n
            out.append(f"    {_YEL}… NOT SHOWN: {rest} more line{'s' if rest > 1 else ''}, "
                       f"{sum(map(len, lines[n:]))} characters{_RST}")
            break
        left -= len(text)
        colour = {"+": "\033[32m", "-": "\033[31m"}.get(line[:1], _DIM)
        out.append(f"    {colour}{text}{_RST}")
    return "\n".join(out)


def _replace(text: str, old: str, new: str, every: bool) -> tuple[int, str]:
    """(how often `old` is in `text`, the text with it replaced). The model was shown
    lines, not line endings: each place keeps the one it has."""
    rx = re.compile(r"\r?\n".join(re.escape(part) for part in old.replace("\r\n", "\n").split("\n")))
    found = [m for m in rx.finditer(text) if m.group()]
    lines = new.replace("\r\n", "\n").split("\n")

    def put(m: re.Match) -> str:
        at = text.find("\n", m.start())  # the ending in it, or of the line it is in
        if at < 0:
            at = text.rfind("\n")
        return ("\r\n" if at > 0 and text[at - 1] == "\r" else "\n").join(lines)

    out, end = [], 0
    for m in found if every else found[:1]:
        out += [text[end:m.start()], put(m)]
        end = m.end()
    return len(found), "".join(out) + text[end:]


def _read_text(p: Path) -> str:
    """A text file as it is. ValueError if it is not one: what cannot be read
    faithfully must not be written back."""
    if p.stat().st_size > _FILE_MAX_BYTES:
        raise ValueError(f"{p.name} is too large ({p.stat().st_size / 1e6:.0f} MB)")
    raw = p.read_bytes()
    if b"\0" in raw[:4096]:
        raise ValueError(f"{p.name} is not a text file")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{p.name} is not UTF-8") from e


class CodeAgent:
    def __init__(self, root: str, base: str, model: str, auto: bool = False,
                 max_rounds: int = 60, quiet: bool = False):
        self.root = Path(root).resolve()
        self.base = base.rstrip("/")
        self.model = model
        self.auto = auto
        self.max_rounds = max_rounds
        self.quiet = quiet
        self.search_cfg = cfgmod.load().search
        # We could ask the engine for web_search/web_fetch, but we send our own
        # tool list, and a turn mixing engine tools with ours would be handed
        # back to us unexecuted. So we carry the same tools ourselves — one
        # implementation, and every call resolves here where the loop sees it.
        self.tools = _FILE_TOOLS + (
            webtools.tool_specs(self.search_cfg) if self.search_cfg.enabled else [])
        self.messages: list[dict] = [{"role": "system", "content": self._system()}]
        self._always: set[str] = set()  # what was answered with "always", this session

    def _system(self) -> str:
        return (
            f"You are a precise coding agent working in the project directory {self.root}.\n\n"
            "Work like this:\n"
            "- Explore before you act: use grep to locate code and read_file to read it. "
            "Never edit a file you have not read.\n"
            "- Prefer edit_file over write_file. Give edit_file an exact, unique old_string "
            "copied from what you read, and change only what the task requires.\n"
            "- Match the surrounding code: its naming, its idioms, its comment density. Do "
            "not add explanatory comments the codebase would not have.\n"
            "- Verify your work by running the project's own tests or build with run_shell. "
            "Read the output — an exit code of 0 from a script that ran no tests proves "
            "nothing. If you did not verify something, say so plainly.\n"
            + ("- Use web_search/web_fetch when you need current documentation or an API you "
               "are unsure about, rather than guessing.\n" if self.search_cfg.enabled else "")
            + "- Keep explanations short. Report what you changed and what you verified.\n\n"
            "All tool paths are relative to the project root; you cannot touch files outside it.")

    # ── path safety ───────────────────────────────────────────────────────────
    def _safe(self, path: str) -> Path:
        p = (self.root / (path or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes the project root: {path}")
        return p

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def _approve(self, what: str, shown: str = "", always: str = "", always_is: str = "") -> bool:
        """Asked of the person, once for `always` if they say so. `shown` is what
        it comes to, in full."""
        if self.auto or (always and always in self._always):
            if not self.quiet:
                print(f"  {_DIM}↳ {_plain(what)}{_RST}")
            return True
        if shown:
            print(shown)
        choices = f"[y]es / [a]lways {always_is} / [N]o" if always else "[y/N]"
        try:
            ans = ask(f"  {_YEL}⚠ {_plain(what)}?{_RST} {choices} ").strip().lower()
        except EOFError:
            print()
            return False
        if always and ans in ("a", "always"):
            self._always.add(always)
            return True
        return ans in ("y", "yes")

    def _walk(self, base: Path):
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in _SKIP_DIRS and not d.startswith(".")]
            for f in fns:
                p = Path(dp) / f
                if p.is_symlink() and self.root not in Path(os.path.realpath(p)).parents:
                    continue  # leads out of the project, or nowhere
                yield p

    # ── tools ─────────────────────────────────────────────────────────────────
    def _t_list_files(self, args: dict) -> str:
        base = self._safe(args.get("path") or ".")
        pattern = args.get("pattern") or ""
        out: list[str] = []
        for p in self._walk(base):
            rel = self._rel(p)
            if pattern and not (fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern)):
                continue
            out.append(rel)
            if len(out) >= _LIST_MAX:
                break
        if not out:
            return "(no matching files)"
        body = "\n".join(sorted(out))
        if len(out) >= _LIST_MAX:
            body += f"\n… truncated at {_LIST_MAX} files — narrow with path or pattern"
        return body

    def _t_grep(self, args: dict) -> str:
        try:
            rx = re.compile(args["pattern"])
        except re.error as e:
            return f"ERROR: bad regular expression: {e}"
        base = self._safe(args.get("path") or ".")
        glob = args.get("glob") or ""
        hits: list[str] = []
        for p in self._walk(base):
            rel = self._rel(p)
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(p.name, glob)):
                continue
            try:
                # Skip binaries cheaply: a NUL byte in the first block is the
                # standard heuristic and avoids decoding a whole object file.
                with p.open("rb") as fh:
                    if b"\0" in fh.read(4096):
                        continue
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(hits) >= _GREP_MAX:
                        break
            if len(hits) >= _GREP_MAX:
                break
        if not hits:
            return "(no matches)"
        body = "\n".join(hits)
        if len(hits) >= _GREP_MAX:
            body += f"\n… truncated at {_GREP_MAX} matches — narrow the pattern"
        return body

    def _t_read_file(self, args: dict) -> str:
        p = self._safe(args["path"])
        if not p.exists():
            return f"ERROR: no such file: {args['path']}"
        if p.is_dir():
            return f"ERROR: {args['path']} is a directory — use list_files"
        if p.stat().st_size > _FILE_MAX_BYTES:
            return f"ERROR: {args['path']} is too large to read — grep it"
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        offset = max(1, int(args.get("offset") or 1))
        limit = max(1, min(_READ_LIMIT_LINES, int(args.get("limit") or _READ_LIMIT_LINES)))
        window = lines[offset - 1: offset - 1 + limit]
        if not window:
            return f"(offset {offset} is past the end — the file has {len(lines)} lines)"
        # Line numbers let the model target edits and refer to code precisely.
        body = "\n".join(f"{offset + i:6d}\t{ln}" for i, ln in enumerate(window))[:_READ_MAX_CHARS]
        tail = ""
        if offset - 1 + len(window) < len(lines):
            tail = (f"\n… {len(lines) - (offset - 1 + len(window))} more lines; "
                    f"read on with offset={offset + len(window)}")
        return body + tail

    def _t_edit_file(self, args: dict) -> str:
        p = self._safe(args["path"])
        if not p.exists():
            return f"ERROR: no such file: {args['path']} — use write_file to create it"
        old, new = args.get("old_string", ""), args.get("new_string", "")
        if not old:
            return "ERROR: old_string is required and must not be empty"
        text = _read_text(p)
        replace_all = bool(args.get("replace_all"))
        count, after = _replace(text, old, new, replace_all)
        if count == 0:
            return ("ERROR: old_string not found. It must match the file EXACTLY, including "
                    "indentation and line breaks. Re-read the file and copy the text verbatim.")
        if count > 1 and not replace_all:
            return (f"ERROR: old_string appears {count} times — the edit would be ambiguous. "
                    "Include surrounding lines to make it unique, or set replace_all.")
        rel = self._rel(p)
        if not self._approve(f"edit {rel} ({count} occurrence{'s' if count > 1 else ''})", _diff(text, after, rel),
                             "edit", "edits"):
            return "DENIED by user"
        if _read_text(p) != text:
            return "ERROR: the file changed meanwhile — read it again"
        p.write_bytes(after.encode("utf-8"))
        return f"edited {rel} ({count if replace_all else 1} replacement(s))"

    def _t_write_file(self, args: dict) -> str:
        p = self._safe(args["path"])
        content = args.get("content", "")
        there = p.exists()
        verb = "overwrite" if there else "create"
        rel = self._rel(p)
        try:
            before = _read_text(p) if there else ""
        except ValueError:
            before = ""  # not a text: it is replaced as a whole, and that is what is asked
        if not self._approve(f"{verb} {rel} ({_fmt_size(len(content))} bytes)", _diff(before, content, rel),
                             "edit", "edits"):
            return "DENIED by user"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("utf-8"))
        return f"{verb}d {rel} ({len(content)} bytes)"

    def _t_run_shell(self, args: dict) -> str:
        cmd = args["command"]
        timeout = max(1, min(600, int(args.get("timeout") or 120)))
        if not self._approve(f"run: {cmd}", always=f"run {cmd}", always_is="this command"):
            return "DENIED by user"
        # In a group of its own, so that all it started ends with it: at the
        # timeout, and at a Ctrl-C (which is for us, not for it).
        p = subprocess.Popen(cmd, shell=True, cwd=self.root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **procs.own_group())
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._end(p.pid)
            return f"ERROR: command timed out after {timeout}s"
        except BaseException:
            self._end(p.pid)
            raise
        body = out.strip()
        if len(body) > _SHELL_MAX_CHARS:
            half = _SHELL_MAX_CHARS // 2
            body = body[:half] + "\n… (output truncated) …\n" + body[-half:]
        return (body or "(no output)") + f"\n[exit {p.returncode}]"

    @staticmethod
    def _end(pid: int) -> None:
        procs.end_tree(pid)
        if os.name == "posix":  # what it left in the background is nobody's child any more, and still of its group
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass

    def _web(self, name: str, args: dict) -> str:
        if name == "web_fetch":
            url = str(args.get("url", "")).strip()
            if "://" not in url:
                url = "http://" + url  # as it is fetched
            try:
                host = urlsplit(url).netloc.lower()
            except ValueError:
                host = ""
            args = {**args, "url": url}
            ok = self._approve(f"fetch {url}", always=f"fetch {host}" if host else "", always_is=f"from {_plain(host)}")
        else:
            ok = self._approve(f"search the web: {args.get('query', '')}", always="search", always_is="search")
        return asyncio.run(webtools.dispatch(name, args, self.search_cfg)) if ok else "DENIED by user"

    def _exec_tool(self, name: str, args: dict) -> str:
        handlers = {
            "list_files": self._t_list_files, "grep": self._t_grep,
            "read_file": self._t_read_file, "edit_file": self._t_edit_file,
            "write_file": self._t_write_file, "run_shell": self._t_run_shell,
        }
        try:
            fn = handlers.get(name)
            if fn is not None:
                return fn(args)
            if name in webtools.TOOL_NAMES:
                return self._web(name, args)
        except KeyError as e:
            return f"ERROR: missing required argument {e}"
        except Exception as e:  # surface tool errors so the model can recover
            return f"ERROR: {e}"
        return f"unknown tool {name!r}"

    # ── context budget ────────────────────────────────────────────────────────
    def _compact(self) -> None:
        """Shrink old tool output when the conversation outgrows the window.

        Ollama's answer to an over-long request is to silently drop the OLDEST
        messages — which are the system prompt and the original task — so the
        agent appears to forget what it was asked. Trimming stale tool results
        ourselves keeps the instructions and recent work intact instead.
        """
        budget = _context_budget(self.model)
        if not budget:
            return
        limit = int(budget * 4 * 0.7)  # ~4 chars/token, leave room to generate
        total = sum(len(str(m.get("content") or "")) for m in self.messages)
        if total <= limit:
            return
        # Oldest first, never the last few messages (that is the live task).
        for m in self.messages[1:-6]:
            if total <= limit:
                break
            if m.get("role") == "tool" and len(m.get("content") or "") > 400:
                was = len(m["content"])
                m["content"] = m["content"][:400] + f"\n… [{was} chars elided to fit context]"
                total -= was - len(m["content"])
        if not self.quiet and total > limit:
            print(f"  {_DIM}(context is full — consider /clear){_RST}")

    # ── one turn ──────────────────────────────────────────────────────────────
    def ask(self, user_text: str) -> str:
        """Run one task to completion. Returns the final assistant text. A Ctrl-C
        ends the task, and leaves a conversation that can be gone on with."""
        self.messages.append({"role": "user", "content": user_text})
        start, pending = len(self.messages), []
        final = ""
        try:
            for _ in range(self.max_rounds):
                self._compact()
                content, calls = self._stream_turn()
                if content is None:  # request failed; message already printed
                    if len(self.messages) == start:
                        self.messages.pop()  # nothing came of it: not asked twice next time
                    return ""
                self.messages.append({"role": "assistant", "content": content or None,
                                      **({"tool_calls": calls} if calls else {})})
                if not calls:
                    final = (content or "").strip()
                    if not final and not self.quiet:
                        print(f"{_DIM}(no reply){_RST}")
                    return final
                pending = list(calls)
                while pending:
                    tc = pending[0]
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = toolloop._parse_args(fn.get("arguments") or "{}")
                    if not self.quiet:
                        print(f"  {_CYAN}→ {_plain(name)}{_RST} {_DIM}{_plain(_tool_label(name, args))}{_RST}")
                    self._result(tc, self._exec_tool(name, args))
                    pending.pop(0)
        except KeyboardInterrupt:
            for tc in pending:
                self._result(tc, "INTERRUPTED by user")
            if self.messages[-1]["role"] != "assistant":
                self.messages.append({"role": "assistant", "content": "(interrupted by the user)"})
            print(f"\n  {_DIM}(interrupted){_RST}")
            return ""
        if not self.quiet:
            print(f"  {_DIM}(stopped after {self.max_rounds} tool rounds — "
                  f"ask me to continue){_RST}")
        return final

    def _result(self, tc: dict, text: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                              "name": (tc.get("function") or {}).get("name", ""), "content": text})

    def _stream_turn(self) -> tuple[str | None, list[dict]]:
        """Stream one assistant turn, printing text as it arrives.

        Streaming is not cosmetic here: a local model chewing through a long
        agentic turn is otherwise a blank terminal for a minute at a time, with
        no way to tell progress from a hang.
        """
        body = {
            "model": self.model, "messages": self.messages, "tools": self.tools,
            "stream": True,
            # We carry web_search/web_fetch ourselves — see __init__.
            "web_search": False,
        }
        acc: dict = {}
        content = ""
        printed = False
        try:
            with httpx.stream("POST", f"{self.base}/v1/chat/completions",
                              json=body, timeout=900.0) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        evt = json.loads(data)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if evt.get("error"):
                        msg = (evt["error"] or {}).get("message", "unknown error")
                        print(f"\n  {_DIM}(backend error: {msg}){_RST}", file=sys.stderr)
                        return None, []
                    delta = ((evt.get("choices") or [{}])[0].get("delta")) or {}
                    if delta.get("tool_calls"):
                        toolloop._accumulate(acc, delta["tool_calls"])
                    if delta.get("content"):
                        content += delta["content"]
                        if not self.quiet:
                            sys.stdout.write(delta["content"])
                            sys.stdout.flush()
                            printed = True
        except httpx.HTTPError as e:
            print(f"\n  {_DIM}(request failed: {e}){_RST}", file=sys.stderr)
            return None, []
        if printed:
            print()
        return content, toolloop._finalize(acc)


def _tool_label(name: str, args: dict) -> str:
    if name == "grep":
        return f"/{args.get('pattern', '')}/ {args.get('glob') or args.get('path') or ''}".strip()
    if name == "run_shell":
        return str(args.get("command", ""))[:100]
    if name == "web_search":
        return str(args.get("query", ""))
    if name == "web_fetch":
        return str(args.get("url", ""))
    return str(args.get("path", ""))


def _context_budget(model: str) -> int:
    """Context window configured for this role/model, 0 if unknown."""
    try:
        cfg = cfgmod.load()
    except Exception:
        return 0
    ctx = cfg.context_for(model)
    if ctx:
        return ctx
    from .ollama_supervisor import effective_default_context
    return effective_default_context()
