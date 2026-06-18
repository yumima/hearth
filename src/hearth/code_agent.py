"""hearth code — a minimal local coding agent.

A tool-calling agent that works on a project directory: the model (bound to the
``coding`` role) can list/read/write files and run shell commands via OpenAI
function-calling against hearth's OWN gateway, looping until the task is done.

It is a lean, in-process alternative to aider/Claude-Code for when you want the
agent to live in hearth itself. File writes and shell commands ask for approval
(unless auto-approve is on), and every path is confined to the project root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

_BOLD, _DIM, _YEL, _RST = "\033[1m", "\033[2m", "\033[33m", "\033[0m"

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List project files (recursively), optionally under a subdirectory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "subdir relative to project root; omit for all"}}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file's contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the FULL new content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command in the project root and return combined stdout+stderr.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".mypy_cache"}


class CodeAgent:
    def __init__(self, root: str, base: str, model: str, auto: bool = False):
        self.root = Path(root).resolve()
        self.base = base.rstrip("/")
        self.model = model
        self.auto = auto
        self.messages: list[dict] = [{"role": "system", "content": self._system()}]

    def _system(self) -> str:
        return (
            f"You are a precise coding agent working in the project directory {self.root}.\n"
            "Use the tools to explore and modify the project. ALWAYS read a file before you edit "
            "it. When you write a file, provide its COMPLETE new content (not a diff). Make "
            "minimal, correct, idiomatic changes that match the surrounding code. Run shell "
            "commands to build or test when it helps verify your work. Keep explanations short. "
            "All tool paths are relative to the project root; never touch files outside it.")

    # ── path safety ───────────────────────────────────────────────────────────
    def _safe(self, path: str) -> Path:
        p = (self.root / (path or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes the project root: {path}")
        return p

    def _approve(self, what: str) -> bool:
        if self.auto:
            print(f"  {_DIM}↳ {what}{_RST}")
            return True
        try:
            ans = input(f"  {_YEL}⚠ {what}?{_RST} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return ans in ("y", "yes")

    # ── tools ───────────────────────────────────────────────────────────────────
    def _exec_tool(self, name: str, args: dict) -> str:
        try:
            if name == "list_files":
                base = self._safe(args.get("path") or ".")
                out: list[str] = []
                for dp, dns, fns in os.walk(base):
                    dns[:] = [d for d in dns if d not in _SKIP_DIRS]
                    for f in fns:
                        out.append(os.path.relpath(os.path.join(dp, f), self.root))
                    if len(out) > 1000:
                        break
                return "\n".join(sorted(out)[:1000]) or "(no files)"
            if name == "read_file":
                return self._safe(args["path"]).read_text(encoding="utf-8", errors="replace")[:60000]
            if name == "write_file":
                p = self._safe(args["path"])
                content = args.get("content", "")
                verb = "overwrite" if p.exists() else "create"
                if not self._approve(f"{verb} {args['path']} ({len(content)} bytes)"):
                    return "DENIED by user"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"{verb}d {args['path']} ({len(content)} bytes)"
            if name == "run_shell":
                cmd = args["command"]
                if not self._approve(f"run: {cmd}"):
                    return "DENIED by user"
                r = subprocess.run(cmd, shell=True, cwd=self.root,
                                   capture_output=True, text=True, timeout=300)
                body = (r.stdout + r.stderr).strip()
                return (body[:20000] or "(no output)") + f"\n[exit {r.returncode}]"
        except Exception as e:  # surface tool errors to the model so it can recover
            return f"ERROR: {e}"
        return f"unknown tool {name!r}"

    # ── one turn (loops over tool calls until a final answer) ───────────────────
    def ask(self, user_text: str) -> None:
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(40):
            resp = self._chat()
            if resp is None:
                return
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            self.messages.append(msg)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                content = (msg.get("content") or "").strip()
                print(content if content else f"{_DIM}(no reply){_RST}")
                return
            for tc in tcs:
                fn = (tc.get("function") or {}).get("name", "")
                try:
                    a = json.loads((tc.get("function") or {}).get("arguments") or "{}")
                except json.JSONDecodeError:
                    a = {}
                label = a.get("path") or a.get("command") or ""
                print(f"  {_DIM}→ {fn} {label}{_RST}")
                result = self._exec_tool(fn, a)
                self.messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                      "content": result})
        print(f"  {_DIM}(stopped after 40 tool rounds — ask me to continue){_RST}")

    def _chat(self) -> dict | None:
        # NB: do NOT force think:false. Reasoning models (qwen3) only emit
        # tool_calls with thinking ENABLED — and reasoning helps coding anyway.
        body = {"model": self.model, "messages": self.messages, "tools": TOOLS,
                "stream": False}
        try:
            r = httpx.post(f"{self.base}/v1/chat/completions", json=body, timeout=600.0)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            print(f"  {_DIM}(request failed: {e}){_RST}", file=sys.stderr)
            return None
