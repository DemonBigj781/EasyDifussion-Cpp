#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ "$(uname -s)" != "Linux" ]; then
    echo "This local Easy Diffusion fork currently supports Linux only." >&2
    exit 1
fi

CONTAINED_ENV="$PROJECT_ROOT/.venv"
PYTHON="$CONTAINED_ENV/bin/python"
CONFIG_PATH="$PROJECT_ROOT/config.yaml"
if [ ! -x "$PYTHON" ]; then
    echo "Easy Diffusion's contained Python environment is missing. Run ./start.sh first." >&2
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Easy Diffusion's configuration is missing: $CONFIG_PATH" >&2
    exit 1
fi

export INSTALL_ENV_DIR="$CONTAINED_ENV"
export EASY_DIFFUSION_CONFIG="$CONFIG_PATH"

exec "$PYTHON" ./scripts/webui_console.py
