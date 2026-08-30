#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SOURCE="$PROJECT_ROOT/source/llama.cpp"
LLAMA_BUILD_DIR="$LLAMA_SOURCE/build"
GGUF_ENV="$PROJECT_ROOT/.venv-llama-cpp"

show_help() {
    cat <<'EOF'
Usage: ./install.sh [options]

Build and install the optional native tools bundled with this fork.

Options:
  --all           Build llama.cpp and install its GGUF converter environment.
  --llama-build   Build llama-cli, llama-server, and llama-quantize.
  --gguf-tools    Install the Python dependencies for Model Tools -> Convert to GGUF.
  --cuda          Require a CUDA-enabled llama.cpp build.
  --cpu           Build llama.cpp without CUDA.
  --jobs N        Set the parallel build job count.
  -h, --help      Show this help.

With no component option, --all is used. CUDA is selected automatically when
both nvcc and nvidia-smi are available. Easy Diffusion itself is started with
./start.sh; this installer only prepares the vendored llama.cpp features.
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

BUILD_LLAMA=false
INSTALL_GGUF=false
CUDA_MODE=auto
JOB_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --all)
            BUILD_LLAMA=true
            INSTALL_GGUF=true
            ;;
        --llama-build)
            BUILD_LLAMA=true
            ;;
        --gguf-tools)
            INSTALL_GGUF=true
            ;;
        --cuda)
            CUDA_MODE=cuda
            ;;
        --cpu)
            CUDA_MODE=cpu
            ;;
        --jobs)
            shift
            [ "$#" -gt 0 ] || fail "--jobs requires a positive integer"
            case "$1" in
                ''|*[!0-9]*) fail "--jobs requires a positive integer" ;;
            esac
            [ "$1" -gt 0 ] || fail "--jobs requires a positive integer"
            JOB_COUNT="$1"
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            fail "Unknown option: $1 (run ./install.sh --help)"
            ;;
    esac
    shift
done

if [ "$BUILD_LLAMA" = false ] && [ "$INSTALL_GGUF" = false ]; then
    BUILD_LLAMA=true
    INSTALL_GGUF=true
fi

[ "$(uname -s)" = "Linux" ] || fail "This local Easy Diffusion fork currently supports Linux only."
[ -f "$PROJECT_ROOT/LICENSE" ] || fail "The project LICENSE file is missing."
[ -f "$PROJECT_ROOT/THIRD_PARTY_NOTICES.md" ] || fail "THIRD_PARTY_NOTICES.md is missing."
[ -f "$LLAMA_SOURCE/LICENSE" ] || fail "The vendored llama.cpp source or its MIT license is missing."

echo "License terms: $PROJECT_ROOT/LICENSE"
echo "Third-party notices: $PROJECT_ROOT/THIRD_PARTY_NOTICES.md"

if [ "$BUILD_LLAMA" = true ]; then
    command -v cmake >/dev/null || fail "cmake is required to build llama.cpp"

    CUDA_ENABLED=OFF
    if [ "$CUDA_MODE" = cuda ]; then
        command -v nvcc >/dev/null || fail "--cuda was requested but nvcc is unavailable"
        command -v nvidia-smi >/dev/null || fail "--cuda was requested but nvidia-smi is unavailable"
        CUDA_ENABLED=ON
    elif [ "$CUDA_MODE" = auto ] && command -v nvcc >/dev/null && command -v nvidia-smi >/dev/null; then
        CUDA_ENABLED=ON
    fi

    GENERATOR_ARGS=()
    if command -v ninja >/dev/null; then
        GENERATOR_ARGS=(-G Ninja)
    fi

    echo "Configuring llama.cpp (GGML_CUDA=$CUDA_ENABLED)..."
    cmake -S "$LLAMA_SOURCE" -B "$LLAMA_BUILD_DIR" "${GENERATOR_ARGS[@]}" \
        -DGGML_CUDA="$CUDA_ENABLED" \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_BUILD_DIR" \
        --target llama-cli llama-server llama-quantize \
        --parallel "$JOB_COUNT"
fi

if [ "$INSTALL_GGUF" = true ]; then
    command -v python3 >/dev/null || fail "Python 3.10-3.14 is required for llama.cpp conversion tools"
    BASE_PYTHON="$(command -v python3)"
    PYTHON_VERSION="$($BASE_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    case "$PYTHON_VERSION" in
        3.10|3.11|3.12|3.13|3.14) ;;
        *) fail "Python 3.10-3.14 is required; found $PYTHON_VERSION at $BASE_PYTHON" ;;
    esac

    if [ ! -x "$GGUF_ENV/bin/python" ]; then
        "$BASE_PYTHON" -m venv "$GGUF_ENV"
    fi
    "$GGUF_ENV/bin/python" -m pip install --upgrade pip wheel
    "$GGUF_ENV/bin/python" -m pip install --editable "$LLAMA_SOURCE/gguf-py"
    "$GGUF_ENV/bin/python" -m pip install \
        --requirement "$LLAMA_SOURCE/requirements/requirements-convert_hf_to_gguf.txt"
    "$GGUF_ENV/bin/python" -c 'import google.protobuf, numpy, sentencepiece, torch, transformers, gguf'
    echo "GGUF conversion environment ready: $GGUF_ENV"
fi

echo "Optional llama.cpp tools are ready."
