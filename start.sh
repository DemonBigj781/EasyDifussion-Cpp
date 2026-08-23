#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ "$(uname -s)" != "Linux" ]; then
    echo "This local Easy Diffusion fork currently supports Linux only." >&2
    exit 1
fi

unset PYTHONHOME

# Bootstrap only the contained Python environment when it is absent. Runtime
# startup never fetches or replaces the checked-in Easy Diffusion source.
if [ ! -x "installer_files/env/bin/python" ]; then
    ./scripts/bootstrap.sh
fi

exec ./scripts/on_env_start.sh
