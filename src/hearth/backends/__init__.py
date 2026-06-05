from .base import Backend, Capabilities, ModelInfo
from .ollama import OllamaBackend

__all__ = ["Backend", "Capabilities", "ModelInfo", "OllamaBackend", "build"]


def build(name: str, btype: str, base_url: str) -> Backend:
    """Construct a backend adapter from config. v1 ships Ollama only."""
    if btype == "ollama":
        return OllamaBackend(name, base_url)
    raise ValueError(f"unsupported backend type {btype!r} (v1 ships: ollama)")
