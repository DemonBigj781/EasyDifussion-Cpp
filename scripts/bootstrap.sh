#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ./scripts/functions.sh

if [ "$(uname -s)" != "Linux" ]; then
    fail "This local Easy Diffusion fork currently supports Linux only."
fi

INSTALL_ENV_DIR="$ROOT_DIR/.venv"
if [ -x "$INSTALL_ENV_DIR/bin/python" ]; then
    exit 0
fi

case "$ROOT_DIR" in
    *" "*) fail "The installation path contains a space, which the contained environment does not support." ;;
esac

PYTHON_REQUIRED=3.13
VENV_ARGS=()
if [ "${EASY_DIFFUSION_XAVIER:-0}" = 1 ]; then
    PYTHON_REQUIRED=3.10
    VENV_ARGS+=(--system-site-packages)
fi
PYTHON_BIN="${EASY_DIFFUSION_PYTHON:-$(command -v python$PYTHON_REQUIRED || true)}"
[ -n "$PYTHON_BIN" ] || fail "Python $PYTHON_REQUIRED is required."
PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$PYTHON_VERSION" = "$PYTHON_REQUIRED" ] || fail "Python $PYTHON_REQUIRED is required; found $PYTHON_VERSION at $PYTHON_BIN."
"$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" "$INSTALL_ENV_DIR" \
    || fail "Could not create the Python $PYTHON_REQUIRED environment."

"$INSTALL_ENV_DIR/bin/python" -m pip install --upgrade pip wheel \
    || fail "Could not initialize pip in Easy Diffusion's Python environment."
