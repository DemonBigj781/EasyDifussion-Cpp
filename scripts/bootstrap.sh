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

PYTHON_BIN="${EASY_DIFFUSION_PYTHON:-$(command -v python3.13 || true)}"
[ -n "$PYTHON_BIN" ] || fail "Python 3.13 is required. Install python3.13 and python3.13-venv first."

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$PYTHON_VERSION" = "3.13" ] || fail "Python 3.13 is required; found $PYTHON_VERSION at $PYTHON_BIN."

"$PYTHON_BIN" -m venv "$INSTALL_ENV_DIR" \
    || fail "Could not create the Python 3.13 environment. Install the python3.13-venv package and retry."

"$INSTALL_ENV_DIR/bin/python" -m pip install --upgrade pip wheel \
    || fail "Could not initialize pip in Easy Diffusion's Python environment."
