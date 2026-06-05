#!/usr/bin/env bash
# Launcher invoked by the hearth.desktop entry. Starts the engine detached if
# it isn't already up, and shows a desktop notification. Idempotent: clicking
# again while it's running just says so. Stop with `hearth stop` (or the
# finterm "Manage local engine" toggle).
set -u

HEARTH_BIN="${1:-${FINCEPT_HEARTH_BIN:-hearth}}"
PORT="${HEARTH_PORT:-11435}"
LOG="$HOME/.hearth/hearth.log"

notify() { command -v notify-send >/dev/null 2>&1 && notify-send "hearth" "$1" || echo "hearth: $1"; }

if curl -fsS "http://127.0.0.1:${PORT}/admin/health" >/dev/null 2>&1; then
    notify "Local AI engine already running on :${PORT}"
    exit 0
fi

if ! command -v "$HEARTH_BIN" >/dev/null 2>&1 && [ ! -x "$HEARTH_BIN" ]; then
    notify "hearth binary not found ($HEARTH_BIN) — reinstall with: pip install hearth"
    exit 1
fi

mkdir -p "$HOME/.hearth"
setsid "$HEARTH_BIN" start >"$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true
notify "Starting local AI engine on :${PORT}… (logs: ~/.hearth/hearth.log)"
