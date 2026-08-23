#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ./scripts/functions.sh

if [ "$(uname -s)" != "Linux" ]; then
    fail "This local Easy Diffusion fork currently supports Linux only."
fi

case "$(uname -m)" in
    x86_64|amd64) MAMBA_ARCH="64" ;;
    arm64|aarch64) MAMBA_ARCH="aarch64" ;;
    *) fail "Unsupported Linux architecture: $(uname -m)" ;;
esac

INSTALL_ENV_DIR="$ROOT_DIR/installer_files/env"
if [ -x "$INSTALL_ENV_DIR/bin/python" ]; then
    exit 0
fi

command -v curl >/dev/null || fail "curl is required to bootstrap Easy Diffusion."
command -v tar >/dev/null || fail "tar is required to bootstrap Easy Diffusion."
command -v bzip2 >/dev/null || fail "bzip2 is required to bootstrap Easy Diffusion."

case "$ROOT_DIR" in
    *" "*) fail "The installation path contains a space, which the contained environment does not support." ;;
esac

MAMBA_ROOT_PREFIX="$ROOT_DIR/installer_files/mamba"
MICROMAMBA="$MAMBA_ROOT_PREFIX/micromamba"
MICROMAMBA_DOWNLOAD_URL="https://micro.mamba.pm/api/micromamba/linux-${MAMBA_ARCH}/latest"

if [ ! -x "$MICROMAMBA" ]; then
    mkdir -p "$MAMBA_ROOT_PREFIX"
    temporary_archive="$(mktemp)"
    trap 'rm -f "$temporary_archive"' EXIT
    curl --fail --location "$MICROMAMBA_DOWNLOAD_URL" --output "$temporary_archive" \
        || fail "Micromamba download failed."
    tar -xvjf "$temporary_archive" -O bin/micromamba > "$MICROMAMBA"
    chmod u+x "$MICROMAMBA"
fi

"$MICROMAMBA" create -y --prefix "$INSTALL_ENV_DIR" -c conda-forge python=3.9 pip \
    || fail "Could not create Easy Diffusion's contained Python environment."
