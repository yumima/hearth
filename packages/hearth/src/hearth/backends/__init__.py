from .base import Backend, Capabilities, ModelInfo
from .ollama import OllamaBackend
from .openai import OpenAIBackend

__all__ = ["Backend", "Capabilities", "ModelInfo", "OllamaBackend", "OpenAIBackend", "build"]


def build(name: str, btype: str, base_url: str, api_key: str = "") -> Backend:
    """Construct a backend adapter from config.

    ``ollama`` — local engine (native /api lifecycle + think control).
    ``openai`` — any OpenAI-compatible remote API (cloud or self-hosted),
    authenticated with ``api_key``.
    """
    if btype == "ollama":
        return OllamaBackend(name, base_url)
    if btype == "openai":
        return OpenAIBackend(name, base_url, api_key)
    raise ValueError(f"unsupported backend type {btype!r} (ships: ollama, openai)")
