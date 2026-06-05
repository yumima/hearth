"""hearth — a stable, OpenAI-compatible local AI engine on loopback.

Fronts Ollama (and later llama.cpp / vLLM / faster-whisper) behind one HTTP
API with a role registry, hardware probe, and tool-call repair. finterm is
its first consumer; the contract is the OpenAI HTTP surface, so any
openai-SDK client works pointed at the base URL.

See finterm's plans/local-ai-engine.md for the full design.
"""

__version__ = "0.1.0"
