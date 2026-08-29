#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ "$(uname -s)" != "Linux" ]; then
    echo "This local Easy Diffusion fork currently supports Linux only." >&2
    exit 1
fi

PYTHON="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Easy Diffusion's contained Python environment is missing. Run ./start.sh first." >&2
    exit 1
fi

exec "$PYTHON" ./scripts/webui_console.py
