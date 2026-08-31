#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SOURCE="$PROJECT_ROOT/source/llama.cpp"
LLAMA_BUILD_DIR="$LLAMA_SOURCE/build"
SDKIT_SOURCE="$PROJECT_ROOT/source/sdkit3-port-source"
GGUF_ENV="$PROJECT_ROOT/.venv"

show_help() {
    cat <<'EOF'
Usage: ./install.sh [options]

Build and install the optional native tools bundled with this fork.

Options:
  --all           Build the native bundle, llama.cpp tools, and GGUF tooling.
  --native-build  Build sdkit/stable-diffusion.cpp with bundled llama-server.
  --llama-build   Build llama-cli, llama-server, and llama-quantize.
  --gguf-tools    Install GGUF conversion tools into Easy Diffusion's main venv.
  --cuda          Require CUDA for the selected native builds.
  --sycl          Build for Intel GPUs with oneAPI/SYCL (source setvars.sh first).
  --cpu           Build llama.cpp without CUDA.
  --jobs N        Set the parallel build job count.
  -h, --help      Show this help.

With no component option, llama.cpp and the GGUF tools are prepared. CUDA is
selected automatically when both nvcc and nvidia-smi are available. Easy
Diffusion itself is started with ./start.sh; this installer prepares the
vendored native runtimes and tools.
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

BUILD_NATIVE=false
BUILD_LLAMA=false
INSTALL_GGUF=false
CUDA_MODE=auto
JOB_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --all)
            BUILD_NATIVE=true
            BUILD_LLAMA=true
            INSTALL_GGUF=true
            ;;
        --native-build)
            BUILD_NATIVE=true
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
        --sycl)
            CUDA_MODE=sycl
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

if [ "$BUILD_NATIVE" = false ] && [ "$BUILD_LLAMA" = false ] && [ "$INSTALL_GGUF" = false ]; then
    BUILD_LLAMA=true
    INSTALL_GGUF=true
fi

[ "$(uname -s)" = "Linux" ] || fail "This local Easy Diffusion fork currently supports Linux only."
[ -f "$PROJECT_ROOT/LICENSE" ] || fail "The project LICENSE file is missing."
[ -f "$PROJECT_ROOT/THIRD_PARTY_NOTICES.md" ] || fail "THIRD_PARTY_NOTICES.md is missing."
[ -f "$LLAMA_SOURCE/LICENSE" ] || fail "The vendored llama.cpp source or its MIT license is missing."
[ -f "$SDKIT_SOURCE/CMakeLists.txt" ] || fail "The vendored sdkit3 source is missing."

echo "License terms: $PROJECT_ROOT/LICENSE"
echo "Third-party notices: $PROJECT_ROOT/THIRD_PARTY_NOTICES.md"

CUDA_ENABLED=OFF
SYCL_ENABLED=OFF
BUILD_PLATFORM=cpu
COMPILER_ARGS=()
if [ "$CUDA_MODE" = cuda ]; then
    command -v nvcc >/dev/null || fail "--cuda was requested but nvcc is unavailable"
    command -v nvidia-smi >/dev/null || fail "--cuda was requested but nvidia-smi is unavailable"
    CUDA_ENABLED=ON
    BUILD_PLATFORM=cuda
elif [ "$CUDA_MODE" = sycl ]; then
    command -v icx >/dev/null || fail "--sycl requires the oneAPI icx compiler (source setvars.sh first)"
    command -v icpx >/dev/null || fail "--sycl requires the oneAPI icpx compiler (source setvars.sh first)"
    command -v sycl-ls >/dev/null || fail "--sycl requires the oneAPI SYCL runtime"
    sycl-ls 2>/dev/null | grep -q '\[level_zero:gpu\]\|\[opencl:gpu\]' ||
        fail "oneAPI does not currently expose an Intel GPU; install/verify the card driver first"
    SYCL_ENABLED=ON
    BUILD_PLATFORM=sycl
    COMPILER_ARGS=(-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
elif [ "$CUDA_MODE" = auto ] && command -v nvcc >/dev/null && command -v nvidia-smi >/dev/null; then
    CUDA_ENABLED=ON
    BUILD_PLATFORM=cuda
fi

GENERATOR_ARGS=()
if command -v ninja >/dev/null; then
    GENERATOR_ARGS=(-G Ninja)
fi

if [ "$BUILD_NATIVE" = true ] || [ "$BUILD_LLAMA" = true ]; then
    command -v cmake >/dev/null || fail "cmake is required to build native runtimes"
fi

if [ "$BUILD_LLAMA" = true ]; then
    echo "Configuring llama.cpp (platform=$BUILD_PLATFORM)..."
    cmake -S "$LLAMA_SOURCE" -B "$LLAMA_BUILD_DIR" "${GENERATOR_ARGS[@]}" \
        "${COMPILER_ARGS[@]}" \
        -DGGML_CUDA="$CUDA_ENABLED" \
        -DGGML_SYCL="$SYCL_ENABLED" \
        -DGGML_SYCL_F16="$SYCL_ENABLED" \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_BUILD_DIR" \
        --target llama-cli llama-server llama-quantize \
        --parallel "$JOB_COUNT"
fi

if [ "$BUILD_NATIVE" = true ]; then
    SDKIT_BUILD_DIR="$SDKIT_SOURCE/build/linux-x64-$BUILD_PLATFORM"
    SDKIT_VARIANT=any
    if [ "$BUILD_PLATFORM" = cuda ]; then
        COMPUTE_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' .')"
        [ -n "$COMPUTE_CAPABILITY" ] || fail "Could not determine the NVIDIA compute capability"
        SDKIT_VARIANT="sm$COMPUTE_CAPABILITY"
    fi
    SDKIT_TARGET_DIR="$PROJECT_ROOT/backends/sdkit3/linux-x64-$BUILD_PLATFORM-$SDKIT_VARIANT"
    echo "Configuring sdkit/stable-diffusion.cpp ($BUILD_PLATFORM) with the integrated llama.cpp runtime..."
    cmake -S "$SDKIT_SOURCE" -B "$SDKIT_BUILD_DIR" "${GENERATOR_ARGS[@]}" \
        "${COMPILER_ARGS[@]}" \
        -DSD_CUDA="$CUDA_ENABLED" \
        -DSD_SYCL="$SYCL_ENABLED" \
        -DGGML_SYCL_F16="$SYCL_ENABLED" \
        -DSDKIT_BUILD_LLAMA_RUNTIME=ON \
        -DSDKIT_LLAMA_BUILD_JOBS="$JOB_COUNT" \
        -DSDKIT_BUILD_NATIVE_VISION=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SDKIT_BUILD_DIR" --target sdkit --parallel "$JOB_COUNT"
    cmake -E make_directory "$SDKIT_TARGET_DIR"
    cmake -E copy_directory "$SDKIT_BUILD_DIR/bin" "$SDKIT_TARGET_DIR"
    echo "Native bundle ready: $SDKIT_TARGET_DIR"
fi

if [ "$INSTALL_GGUF" = true ]; then
    [ -x "$GGUF_ENV/bin/python" ] || fail "The main Easy Diffusion venv is missing. Run ./start.sh once first."
    PYTHON_VERSION="$($GGUF_ENV/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [ "$PYTHON_VERSION" = "3.13" ] || fail "Easy Diffusion's main venv must use Python 3.13; found $PYTHON_VERSION."

    "$GGUF_ENV/bin/python" -m pip install --upgrade pip wheel
    "$GGUF_ENV/bin/python" -m pip install --editable "$LLAMA_SOURCE/gguf-py"
    "$GGUF_ENV/bin/python" -m pip install \
        "numpy>=2.1,<2.3" \
        "sentencepiece>=0.1.98,<0.3" \
        "transformers==4.57.6" \
        "protobuf>=4.21,<5"
    "$GGUF_ENV/bin/python" -c 'import torch; assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 6)'
    "$GGUF_ENV/bin/python" -c 'import google.protobuf, numpy, sentencepiece, torch, transformers, gguf'
    echo "GGUF conversion tools are ready in: $GGUF_ENV"
fi

echo "Optional llama.cpp tools are ready."
