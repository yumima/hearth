"""OpenAI ↔ Ollama-native message translation.

Every case here is a shape that produces a hard 400 (or silently drops data) if
passed through unchanged — the native API differs from the OpenAI one by field
*type*, not just by name, and nothing catches that until a real request fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hearth.backends.ollama import OllamaBackend  # noqa: E402

_to_native = OllamaBackend._to_native_messages
_from_native = OllamaBackend._native_tool_calls


def test_assistant_tool_arguments_string_becomes_object():
    """The bug that broke replaying our own tool history: native parses
    `arguments` as an object and 400s on a JSON string."""
    out = _to_native([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "web_search", "arguments": '{"query":"x"}'}}]}])
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {"query": "x"}, "arguments must be an object, not a string"
    assert out[0]["content"] == "", "native rejects null content"


def test_malformed_arguments_degrade_to_empty_object():
    out = _to_native([{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "f", "arguments": "{broken"}}]}])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {}


def test_tool_result_uses_tool_name_not_name():
    out = _to_native([{"role": "tool", "tool_call_id": "c1",
                       "name": "web_search", "content": "result"}])
    assert out[0]["tool_name"] == "web_search"
    assert "name" not in out[0]


def test_multimodal_parts_become_text_plus_base64_images():
    out = _to_native([{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]}])
    assert out[0]["content"] == "what is this?"
    assert out[0]["images"] == ["QUJD"], "the data: prefix must be stripped"


def test_plain_string_content_is_untouched():
    out = _to_native([{"role": "user", "content": "hello"}])
    assert out == [{"role": "user", "content": "hello"}]


def test_native_tool_calls_convert_back_to_openai_string_arguments():
    calls = _from_native({"tool_calls": [
        {"function": {"name": "web_search", "arguments": {"query": "x"}}}]})
    assert isinstance(calls[0]["function"]["arguments"], str)
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "x"}
    assert calls[0]["id"], "native carries no id; one must be synthesized"
    assert calls[0]["type"] == "function"


def test_round_trip_survives_native_then_openai_then_native():
    """The tool loop replays its own history, so translation must be stable."""
    native = _from_native({"tool_calls": [
        {"function": {"name": "web_fetch", "arguments": {"url": "https://e.com"}}}]})
    back = _to_native([{"role": "assistant", "content": "",
                        "tool_calls": [{k: v for k, v in native[0].items() if k != "index"}]}])
    assert back[0]["tool_calls"][0]["function"]["arguments"] == {"url": "https://e.com"}


def test_num_ctx_reaches_native_options():
    """A per-role context window is the whole reason the native path exists for
    non-thinking requests."""
    b = OllamaBackend.__new__(OllamaBackend)  # no client needed for body building
    body = b._to_native({"model": "m", "messages": [], "num_ctx": 16384}, False, False)
    assert body["options"]["num_ctx"] == 16384


def test_absent_num_ctx_leaves_options_alone():
    b = OllamaBackend.__new__(OllamaBackend)
    body = b._to_native({"model": "m", "messages": []}, False, False)
    assert "options" not in body
