# hearth + mantel — work on both from a checkout (uv: https://docs.astral.sh/uv).
#
#   make build     both, each in an environment of its own
#   make test      both packages' tests, as CI runs them
#   make start     the gateway (foreground; Ctrl-C to stop)      make stop
#   make status    is it up? show roles
#   make chat      terminal chat                                 make client   mantel's window
#   make lock      after changing a dependency
#
# Ollama (the inference backend) is fetched into vendor/ separately — see README.md.

HEARTH := .venv/bin/hearth
MANTEL := .venv-mantel/bin/mantel

.PHONY: build test start stop status chat client models hardware lock install-cli uninstall-cli

build:
	uv sync --frozen --package hearth --extra dev --extra tts --extra stt
	UV_PROJECT_ENVIRONMENT=.venv-mantel uv sync --frozen --package mantel --extra dev --extra gui
	.venv/bin/python -c "from pathlib import Path; from hearth import client; Path('.venv-mantel/.complete').write_text(client._locked(Path('.')))"
	@git rev-parse --git-dir >/dev/null 2>&1 && git config core.hooksPath .githooks || true

test:
	.venv/bin/python -m pytest -q packages/hearth tests
	.venv-mantel/bin/python -m pytest -q packages/mantel

lock:    ; uv lock
start:   ; $(HEARTH) start
stop:    ; $(HEARTH) stop
status:  ; $(HEARTH) status
chat:    ; $(HEARTH) chat
client:  ; $(MANTEL)
models:  ; $(HEARTH) models
hardware:; $(HEARTH) hardware

# Both commands on PATH (symlinks in ~/.local/bin).
install-cli:
	mkdir -p $$HOME/.local/bin
	ln -sf $(abspath $(HEARTH)) $$HOME/.local/bin/hearth
	ln -sf $(abspath $(MANTEL)) $$HOME/.local/bin/mantel
uninstall-cli: ; rm -f $$HOME/.local/bin/hearth $$HOME/.local/bin/mantel
