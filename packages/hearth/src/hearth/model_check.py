"""Does a model work here — how fast, how much of it on the GPU, and does it do
what a chat program needs of it? Against an Ollama, with the context it will have.

Nothing here is about any one program: JSON when asked for, a tool called with
the right arguments, figures copied as given, and a fact found again in a long
text. A model that fails one of those makes every program using it look broken.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import httpx

from .catalog import CONTEXT_STEPS, LEAST_CONTEXT

# Below this a chat is too slow to use; the plan then suggests a smaller model.
SLOW_GPU, SLOW_CPU = 8.0, 2.0
_ON_GPU = 0.995   # Ollama's own two numbers differ by rounding
_TIMEOUT = 600.0


@dataclass
class Result:
    context: int = 0
    gpu_share: float = 0.0          # of the model, 1 = all of it
    tokens_per_second: float = 0.0
    load_seconds: float = 0.0
    failed: list[str] = field(default_factory=list)  # checks, in words
    error: str = ""                 # it could not be run at all

    @property
    def ok(self) -> bool:
        return not self.error and not self.failed

    def as_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


class Stopped(Exception):
    """The person asked to stop."""


class Checker:
    def __init__(self, base_url: str, model: str, stopped: Callable[[], bool] = lambda: False,
                 say: Callable[[str], None] = lambda s: None):
        self.base, self.model = base_url.rstrip("/"), model
        self._stopped, self._say = stopped, say
        self._http = httpx.Client(timeout=httpx.Timeout(_TIMEOUT, connect=5.0))

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, body: dict) -> dict:
        if self._stopped():
            raise Stopped
        r = self._http.post(self.base + path, json=body)
        data = r.json() if r.content else {}
        if r.status_code != 200:
            raise RuntimeError(str(data.get("error") or f"HTTP {r.status_code}"))
        return data

    def _chat(self, messages: list[dict], context: int, predict: int, **more) -> dict:
        return self._post("/api/chat", {
            "model": self.model, "messages": messages, "stream": False, "think": False, "keep_alive": "5m",
            "options": {"num_ctx": context, "num_predict": predict, "temperature": 0, "seed": 7}, **more})

    def _chat_any(self, messages: list[dict], context: int, predict: int, **more) -> dict:
        """A model that cannot switch its thinking off refuses the field."""
        try:
            return self._chat(messages, context, predict, **more)
        except RuntimeError as e:
            if "think" not in str(e).lower():
                raise
            return self._post("/api/chat", {
                "model": self.model, "messages": messages, "stream": False, "keep_alive": "5m",
                "options": {"num_ctx": context, "num_predict": predict + 600, "temperature": 0, "seed": 7}, **more})

    # ── loading ──────────────────────────────────────────────────────────────
    def _share(self) -> float:
        for m in self._http.get(self.base + "/api/ps").json().get("models", []):
            if m.get("name") in (self.model, self.model + ":latest"):
                return (m.get("size_vram") or 0) / m["size"] if m.get("size") else 0.0
        return 0.0

    def unload(self) -> None:
        try:
            self._http.post(self.base + "/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass

    def load(self, context: int, on_gpu: bool) -> tuple[int, float, float]:
        """(context, share on the GPU, seconds). On a GPU the context is lowered
        until all of it is there; the least one is kept even if it is not."""
        steps = [s for s in CONTEXT_STEPS if LEAST_CONTEXT <= s <= context] if on_gpu else [context]
        steps = steps or [context]
        for i, c in enumerate(steps):
            self._say(f"Loading it with room for {c} tokens…")
            r = self._chat_any([{"role": "user", "content": "Say OK."}], c, 4)
            share = self._share()
            if not on_gpu or share >= _ON_GPU or i == len(steps) - 1:
                return c, share, (r.get("load_duration") or 0) / 1e9
        raise AssertionError

    # ── what it has to be able to do ─────────────────────────────────────────
    def _json(self, c: int) -> bool:
        r = self._chat_any([{"role": "user", "content":
                             "Reply with a JSON object with the keys \"city\" (string) and \"population\" (number) "
                             "for a city with 120000 people called Lund."}], c, 120, format="json")
        try:
            d = json.loads(r["message"]["content"])
        except (ValueError, KeyError, TypeError):
            return False
        return isinstance(d, dict) and d.get("city") == "Lund" and d.get("population") == 120000

    def _tool(self, c: int) -> bool:
        r = self._chat_any(
            [{"role": "user", "content": "What is the weather in Oslo, in celsius? Use the tool."}], c, 200,
            tools=[{"type": "function", "function": {
                "name": "get_weather", "description": "Current weather in a city",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                    "required": ["city", "unit"]}}},
                {"type": "function", "function": {
                    "name": "send_mail", "description": "Send an e-mail",
                    "parameters": {"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]}}}])
        calls = r.get("message", {}).get("tool_calls") or []
        f = calls[0].get("function", {}) if len(calls) == 1 else {}
        args = f.get("arguments")
        args = json.loads(args) if isinstance(args, str) else args
        return (f.get("name") == "get_weather" and isinstance(args, dict)
                and "oslo" in str(args.get("city", "")).lower() and args.get("unit") == "celsius")

    def _figures(self, c: int) -> bool:
        r = self._chat_any([{"role": "user", "content":
                             "Revenue was 4,817.36 million and the margin 23.45%. Earnings per share were -0.072. "
                             "Repeat the three figures exactly as given, separated by semicolons, and nothing else."}],
                           c, 80)
        # (Without the thousands' comma it is the same figure; with a minus sign of
        # another kind, too.)
        text = re.sub(r"[\s,]", "", r.get("message", {}).get("content", "")).replace("\u2212", "-")
        return all(x in text for x in ("4817.36", "23.45%", "-0.072"))

    def _long(self, c: int) -> bool:
        """A fact two thirds of the way into a text that fills most of the context."""
        lines = [f"Entry {i}: the shipment of crate {i * 37 % 991} was recorded at gate {i % 9 + 1}." for i in
                 range(max(40, min(c, 16384) * 5 // 8 // 22))]
        lines.insert(len(lines) * 2 // 3, "Entry X: the vault code is 58213.")
        r = self._chat_any([{"role": "user", "content": "\n".join(lines) + "\n\nWhat is the vault code? Answer with the number only."}],
                           c, 30)
        return "58213" in r.get("message", {}).get("content", "")

    CHECKS = (("it does not answer in JSON when asked to", "_json"),
              ("it does not use tools correctly", "_tool"),
              ("it does not copy figures exactly", "_figures"),
              ("it loses facts in a long text", "_long"))

    def _speed(self, c: int) -> float:
        r = self._chat_any([{"role": "user", "content": "Write three sentences about the sea."}], c, 120)
        n, t = r.get("eval_count") or 0, (r.get("eval_duration") or 0) / 1e9
        return n / t if t else 0.0

    def run(self, context: int, on_gpu: bool) -> Result:
        res = Result()
        try:
            res.context, res.gpu_share, res.load_seconds = self.load(context, on_gpu)
            self._say("Measuring its speed…")
            res.tokens_per_second = round(self._speed(res.context), 1)
            for words, name in self.CHECKS:
                if name == "_long" and not on_gpu:
                    continue  # minutes on a processor, for an answer known from the model itself
                self._say("Checking what it can do…")
                try:
                    passed = getattr(self, name)(res.context)
                except (RuntimeError, httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
                    passed = False  # refused ("does not support tools"), or no answer in time
                if not passed:
                    res.failed.append(words)
        except Stopped:
            raise
        except (RuntimeError, httpx.HTTPError, ValueError, KeyError) as e:
            res.error = str(e) or type(e).__name__
        return res
