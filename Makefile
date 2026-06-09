# hearth — build / run / manage the local AI engine.
#
#   make build         set up the venv + install hearth (editable)
#   make start         start the gateway (foreground; Ctrl-C to stop)
#   make stop          stop a running gateway
#   make status        is it up? show roles
#   make chat          interactive terminal chat with the local engine
#   make models        list servable models
#   make hardware      GPU / VRAM / RAM probe
#   make test          run unit tests
#   make install-client   install the clickable Hearth chat desktop app
#   make uninstall-client remove the desktop app launcher
#
# Ollama (the inference backend) is fetched into vendor/ separately — see
# README.md. `make start` brings up whatever Ollama it finds.

VENV   := .venv
PIP    := $(VENV)/bin/pip
HEARTH := $(VENV)/bin/hearth
PYTEST := $(VENV)/bin/python -m pytest

.PHONY: build start stop status chat models hardware test install-client uninstall-client

build:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@git rev-parse --git-dir >/dev/null 2>&1 && git config core.hooksPath .githooks \
		&& echo "✓ pre-commit changelog hook wired (.githooks)" || true
	@echo "✓ built. Next: 'make start' (then 'make chat' in another shell)."

start:   ; $(HEARTH) start
stop:    ; $(HEARTH) stop
status:  ; $(HEARTH) status
chat:    ; $(HEARTH) chat
models:  ; $(HEARTH) models
hardware:; $(HEARTH) hardware
test:    ; $(PYTEST) -q

install-client:   ; $(HEARTH) client install
uninstall-client: ; $(HEARTH) client uninstall

# Put `hearth` on PATH (symlink into ~/.local/bin) so you can run it from any
# shell without activating the venv. (~/.local/bin is on PATH on most distros.)
install-cli:
	mkdir -p $$HOME/.local/bin
	ln -sf $(abspath $(HEARTH)) $$HOME/.local/bin/hearth
	@echo "✓ linked $$HOME/.local/bin/hearth -> $(abspath $(HEARTH))  (open a new shell or 'hash -r')"
uninstall-cli: ; rm -f $$HOME/.local/bin/hearth; @echo "✓ removed ~/.local/bin/hearth"
.PHONY: install-cli uninstall-cli
