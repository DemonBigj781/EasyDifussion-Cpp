#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/sdkit3-port-source"
CUDA_ARCH="${SDKIT_CUDA_ARCHITECTURES:-86}"
BUILD_TYPE="${SDKIT_BUILD_TYPE:-Release}"
BUILD_JOBS="${SDKIT_BUILD_JOBS:-4}"
TARGET="${SDKIT_BUILD_TARGET:-sdkit}"
BUILD_DIR="${SDKIT_BUILD_DIR:-$PROJECT_DIR/build/local-linux-x64-cuda-sm${CUDA_ARCH}}"
USE_CUDA=ON
DEPLOY=0
DEPLOY_DIR="${SDKIT_DEPLOY_DIR:-}"

usage() {
    cat <<'EOF'
Usage: source/build.sh [options]

Options:
  --cuda-arch N     CUDA architecture, such as 86 (default: 86)
  --cpu             Build without CUDA
  --build-dir PATH  Build outside the default local build directory
  --build-type TYPE CMake build type (default: Release)
  --jobs N          Parallel build jobs (default: 4)
  --target NAME     CMake target (default: sdkit)
  --deploy PATH     Copy the completed bin bundle to PATH
  -h, --help        Show this help

Environment equivalents: SDKIT_CUDA_ARCHITECTURES, SDKIT_BUILD_DIR,
SDKIT_BUILD_TYPE, SDKIT_BUILD_JOBS, SDKIT_BUILD_TARGET, SDKIT_DEPLOY_DIR.
Setting SDKIT_DEPLOY_DIR alone does not deploy; --deploy is required.
EOF
}

while (($#)); do
    case "$1" in
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --cpu)
            USE_CUDA=OFF
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --jobs)
            BUILD_JOBS="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --deploy)
            DEPLOY=1
            DEPLOY_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$PROJECT_DIR/CMakeLists.txt" ]]; then
    echo "sdkit3 source tree not found: $PROJECT_DIR" >&2
    exit 1
fi
if [[ ! "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer" >&2
    exit 2
fi
if [[ "$USE_CUDA" == ON && ! "$CUDA_ARCH" =~ ^[0-9]+([;][0-9]+)*$ ]]; then
    echo "--cuda-arch must be a number or semicolon-separated list" >&2
    exit 2
fi

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    CACHED_SOURCE="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "$BUILD_DIR/CMakeCache.txt")"
    if [[ -n "$CACHED_SOURCE" && "$CACHED_SOURCE" != "$PROJECT_DIR" ]]; then
        echo "Build cache belongs to a different source tree: $CACHED_SOURCE" >&2
        echo "Choose another directory with --build-dir; no cache was deleted." >&2
        exit 1
    fi
fi

CMAKE_ARGS=(
    -S "$PROJECT_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DSD_CUDA="$USE_CUDA"
    -DGGML_NATIVE=OFF
    -DSDKIT_BUILD_NATIVE_VISION=OFF
    -DSDKIT_BUILD_IMAGE_TOOLS=ON
)
if [[ "$USE_CUDA" == ON ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target "$TARGET" --parallel "$BUILD_JOBS"

BIN_DIR="$BUILD_DIR/bin"
if [[ "$DEPLOY" == 1 ]]; then
    if [[ -z "$DEPLOY_DIR" || "$DEPLOY_DIR" == "/" || "$DEPLOY_DIR" == "$SCRIPT_DIR" ]]; then
        echo "Refusing unsafe deployment path: ${DEPLOY_DIR:-<empty>}" >&2
        exit 2
    fi
    mkdir -p "$DEPLOY_DIR"
    cmake -E copy_directory "$BIN_DIR" "$DEPLOY_DIR"
    echo "Deployed native bundle to $DEPLOY_DIR"
fi

echo "Built $TARGET in $BIN_DIR"
