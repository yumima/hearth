# The engine and the chat window — work on both from a checkout (uv: https://docs.astral.sh/uv).
#
#   make build     both, each in an environment of its own
#   make test      both packages' tests, as CI runs them
#   make start     the engine                                    make stop
#   make status    is it up? show roles
#   make chat      terminal chat                                 make client   the window
#   make lock      after changing a dependency
#
# Ollama (the inference backend) is fetched into vendor/ separately — see README.md.

HEARTH := .venv/bin/hearth

.PHONY: build test start stop status chat client models lock install-cli uninstall-cli

build:
	uv sync --frozen --package hearth --extra dev --extra tts --extra stt
	UV_PROJECT_ENVIRONMENT=.venv-client uv sync --frozen --package hearth-client --extra dev --extra gui
	@git rev-parse --git-dir >/dev/null 2>&1 && git config core.hooksPath .githooks || true

test:
	.venv/bin/python -m pytest -q packages/hearth tests
	.venv-client/bin/python -m pytest -q packages/client

lock:    ; uv lock
start:   ; $(HEARTH) start
stop:    ; $(HEARTH) stop
status:  ; $(HEARTH) status
chat:    ; $(HEARTH) chat
client:  ; $(HEARTH) client
models:  ; $(HEARTH) model list

# The command on PATH (a symlink in ~/.local/bin).
install-cli:
	mkdir -p $$HOME/.local/bin
	ln -sf $(abspath $(HEARTH)) $$HOME/.local/bin/hearth
uninstall-cli: ; rm -f $$HOME/.local/bin/hearth
