#!/usr/bin/env bash
# Remove the hearth desktop launcher installed by install-app.sh.
set -euo pipefail

APPS_DIR="$HOME/.local/share/applications"
DATA_DIR="$HOME/.local/share/hearth"
DESKTOP_FILE="$APPS_DIR/hearth.desktop"

rm -f "$DESKTOP_FILE"
rm -rf "$DATA_DIR"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" 2>/dev/null || true

echo "✓ removed the hearth desktop launcher."
echo "  (This does not uninstall the hearth package or stop a running engine.)"
