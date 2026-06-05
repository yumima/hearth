#!/usr/bin/env bash
# Install a clickable "hearth" desktop launcher (Linux / freedesktop).
# Resolves the hearth binary, installs the launcher, and writes a .desktop
# entry into ~/.local/share/applications so hearth shows up in your app menu.
#
# Reverse with: scripts/uninstall-app.sh  (or `make uninstall-app`).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$HOME/.local/share/applications"
DATA_DIR="$HOME/.local/share/hearth"
DESKTOP_FILE="$APPS_DIR/hearth.desktop"
LAUNCHER="$DATA_DIR/hearth-app-launch.sh"

# Resolve the hearth binary: env override > pip/pipx > dev venv > PATH.
resolve_hearth() {
    if [ -n "${FINCEPT_HEARTH_BIN:-}" ] && [ -x "$FINCEPT_HEARTH_BIN" ]; then echo "$FINCEPT_HEARTH_BIN"; return; fi
    for c in "$HOME/.local/bin/hearth" "$REPO_DIR/.venv/bin/hearth" "$HOME/.hearth/bin/hearth"; do
        [ -x "$c" ] && { echo "$c"; return; }
    done
    command -v hearth 2>/dev/null || true
}

HEARTH_BIN="$(resolve_hearth)"
if [ -z "$HEARTH_BIN" ]; then
    echo "error: hearth binary not found. Install it first (e.g. 'make build' or 'pip install hearth')," >&2
    echo "       or set FINCEPT_HEARTH_BIN to its path, then re-run." >&2
    exit 1
fi
echo "hearth binary: $HEARTH_BIN"

mkdir -p "$APPS_DIR" "$DATA_DIR"
install -m 0755 "$REPO_DIR/scripts/hearth-app-launch.sh" "$LAUNCHER"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=hearth
Comment=Start the local AI engine (OpenAI-compatible, on 127.0.0.1:11435)
Exec="$LAUNCHER" "$HEARTH_BIN"
Icon=applications-science
Terminal=false
Categories=Development;
Keywords=ai;llm;ollama;local;engine;
EOF
chmod 0644 "$DESKTOP_FILE"

# Refresh the menu cache (best-effort).
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" 2>/dev/null || true

echo "✓ installed: $DESKTOP_FILE"
echo "  Search your app menu for 'hearth' and click to start the engine."
echo "  Stop it with: hearth stop   (or the finterm 'Manage local engine' toggle)"
