#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ./scripts/functions.sh

if [ "$(uname -s)" != "Linux" ]; then
    fail "This local Easy Diffusion fork currently supports Linux only."
fi

printf "\n\nEasy Diffusion - local Linux fork\n\n"
export PYTHONNOUSERSITE=y

if [ -f "scripts/config.sh" ]; then
    source scripts/config.sh
fi

if [ -f "scripts/user_config.sh" ]; then
    source scripts/user_config.sh
fi

if [ ! -d "ui/easydiffusion" ]; then
    fail "The checked-in Easy Diffusion UI is missing: ui/easydiffusion"
fi

# ui/ and scripts/ in this repository are authoritative. Do not copy, fetch,
# reset, or otherwise replace them during startup.
exec ./scripts/on_sd_start.sh
