#!/usr/bin/env bash
set -euo pipefail

# Build script: auto-detects available bundlers
# - CI (GitHub Actions): builds all targets (appimagetool is installed)
# - Local: builds only targets whose tools are available

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

TARGETS=""

# Check each bundler
command -v dpkg-deb &>/dev/null && TARGETS="${TARGETS:+$TARGETS,}deb"
command -v rpmbuild &>/dev/null && TARGETS="${TARGETS:+$TARGETS,}rpm"
command -v appimagetool &>/dev/null && TARGETS="${TARGETS:+$TARGETS,}appimage"

# macOS
command -v hdiutil &>/dev/null && TARGETS="${TARGETS:+$TARGETS,}dmg"

# Windows (cross-compile or native)
[[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]] && TARGETS="${TARGETS:+$TARGETS,}nsis"

# Fallback: at least deb
TARGETS="${TARGETS:-deb}"

echo "Building bundles: $TARGETS"

# Node 20 for Vite 7
if [[ -d "/tmp/node-v20.19.0-linux-x64/bin" ]]; then
  export PATH="/tmp/node-v20.19.0-linux-x64/bin:$PATH"
fi

exec npx tauri build --bundles "$TARGETS" "$@"
