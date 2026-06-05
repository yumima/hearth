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
#   make install-app   install a clickable desktop launcher (Linux)
#   make uninstall-app remove the desktop launcher
#
# Ollama (the inference backend) is fetched into vendor/ separately — see
# README.md. `make start` brings up whatever Ollama it finds.

VENV   := .venv
PIP    := $(VENV)/bin/pip
HEARTH := $(VENV)/bin/hearth
PYTEST := $(VENV)/bin/python -m pytest

.PHONY: build start stop status chat models hardware test install-app uninstall-app

build:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@echo "✓ built. Next: 'make start' (then 'make chat' in another shell)."

start:   ; $(HEARTH) start
stop:    ; $(HEARTH) stop
status:  ; $(HEARTH) status
chat:    ; $(HEARTH) chat
models:  ; $(HEARTH) models
hardware:; $(HEARTH) hardware
test:    ; $(PYTEST) -q

install-app:   ; bash scripts/install-app.sh
uninstall-app: ; bash scripts/uninstall-app.sh
