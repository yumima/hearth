"""hearth — a stable, OpenAI-compatible local AI engine on loopback.

Fronts Ollama (and later llama.cpp / vLLM / faster-whisper) behind one HTTP
API with a role registry, hardware probe, and tool-call repair. finterm is
its first consumer; the contract is the OpenAI HTTP surface, so any
openai-SDK client works pointed at the base URL.

See finterm's plans/local-ai-engine.md for the full design.
"""

__version__ = "0.1.0"

# Consumer-facing API contract version. Bump the MAJOR when a consumer
# (finterm) would need code changes; the MINOR for additive, back-compatible
# surface. Consumers pin a floor against this (the Qt-6.8-floor analogue),
# read from GET /admin/version.
CONTRACT_VERSION = "0.1"
