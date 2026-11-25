#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export EASY_DIFFUSION_XAVIER=1
export EASY_DIFFUSION_PYTHON="${EASY_DIFFUSION_PYTHON:-$(command -v python3.10 || true)}"
export PIP_ONLY_BINARY="${PIP_ONLY_BINARY:-:all:}"
exec ./start.sh
