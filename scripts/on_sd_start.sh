#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ./scripts/functions.sh

if [ "$(uname -s)" != "Linux" ]; then
    fail "This local Easy Diffusion fork currently supports Linux only."
fi

export INSTALL_ENV_DIR="$ROOT_DIR/installer_files/env"
PYTHON="$INSTALL_ENV_DIR/bin/python"

if [ ! -x "$PYTHON" ]; then
    fail "The contained Python environment is missing: $PYTHON"
fi

export PATH="$INSTALL_ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=y
unset PYTHONHOME

# Keep runtime dependencies self-contained, then launch the checked-in UI.
"$PYTHON" ./scripts/ensure_torchruntime.py
exec "$PYTHON" ./scripts/check_modules.py --launch-uvicorn
