#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/sdkit3-port-source"
CUDA_ARCH="${SDKIT_CUDA_ARCHITECTURES:-86}"
CUDA_VERSION="${SDKIT_CUDA_VERSION:-}"
SYCL_DEVICE_ARCH="${SDKIT_SYCL_DEVICE_ARCH:-}"
BUILD_TYPE="${SDKIT_BUILD_TYPE:-Release}"
BUILD_JOBS="${SDKIT_BUILD_JOBS:-4}"
TARGET="${SDKIT_BUILD_TARGET:-sdkit}"
BUILD_DIR="${SDKIT_BUILD_DIR:-}"
BUILD_DIR_EXPLICIT=0
[[ -n "$BUILD_DIR" ]] && BUILD_DIR_EXPLICIT=1
USE_CUDA=ON
USE_SYCL=OFF
USE_VULKAN=OFF
XAVIER_PRESET=""
DEPLOY=0
DEPLOY_DIR="${SDKIT_DEPLOY_DIR:-}"

usage() {
    cat <<'EOF'
Usage: source/Build.sh [options]

Options:
  --cuda-arch N     CUDA architecture, such as 86 (default: 86)
  --cuda-VERSION    Select /usr/local/cuda-VERSION (for example --cuda-11.4)
  --cuda-version V  Select /usr/local/cuda-V
  --cpu             Build the CPU-only backend
  --sycl            Build for Intel GPUs with oneAPI/SYCL
  --sycl-device-arch ARCH
                    Ahead-of-time Intel GPU architecture (default: JIT)
  --vulkan          Build the portable Vulkan GPU backend
  --xavier          Jetson Xavier, NVIDIA-supplied JetPack 5 preset
  --xavier20        Jetson Xavier, custom Ubuntu 20.04 preset
  --xavier22        Jetson Xavier, custom Ubuntu 22.04 preset
  --build-dir PATH  Build outside the default local build directory
  --build-type TYPE CMake build type (default: Release)
  --jobs N          Parallel build jobs (default: 4)
  --target NAME     CMake target (default: sdkit)
  --deploy PATH     Copy the completed bin bundle to PATH
  -h, --help        Show this help

Environment equivalents: SDKIT_CUDA_ARCHITECTURES, SDKIT_CUDA_VERSION,
SDKIT_SYCL_DEVICE_ARCH,
SDKIT_BUILD_DIR, SDKIT_BUILD_TYPE, SDKIT_BUILD_JOBS, SDKIT_BUILD_TARGET,
SDKIT_DEPLOY_DIR.
Setting SDKIT_DEPLOY_DIR alone does not deploy; --deploy is required.
EOF
}

while (($#)); do
    case "$1" in
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --cuda-[0-9]*)
            CUDA_VERSION="${1#--cuda-}"
            shift
            ;;
        --cpu)
            USE_CUDA=OFF
            USE_SYCL=OFF
            USE_VULKAN=OFF
            shift
            ;;
        --sycl)
            USE_CUDA=OFF
            USE_SYCL=ON
            USE_VULKAN=OFF
            shift
            ;;
        --sycl-device-arch)
            SYCL_DEVICE_ARCH="$2"
            shift 2
            ;;
        --vulkan)
            USE_CUDA=OFF
            USE_SYCL=OFF
            USE_VULKAN=ON
            shift
            ;;
        --xavier)
            XAVIER_PRESET="jetpack5"
            shift
            ;;
        --xavier20)
            XAVIER_PRESET="ubuntu20"
            shift
            ;;
        --xavier22)
            XAVIER_PRESET="ubuntu22"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            BUILD_DIR_EXPLICIT=1
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

# Platform presets are intentionally applied after parsing. Consequently they
# win over --cpu, --cuda-arch, and their environment equivalents regardless of
# command-line order. Output/deployment and build orchestration options remain
# user-selectable because they do not change the generated machine code.
if [[ -n "$XAVIER_PRESET" ]]; then
    USE_CUDA=ON
    USE_SYCL=OFF
    USE_VULKAN=OFF
    CUDA_ARCH=72
    # NVIDIA's stock JetPack 5 Xavier image ships CUDA 11.4. An explicit
    # --cuda-VERSION (or SDKIT_CUDA_VERSION) remains available for a custom
    # toolkit while all other Xavier platform constraints still apply.
    if [[ "$XAVIER_PRESET" == "jetpack5" && -z "$CUDA_VERSION" ]]; then
        CUDA_VERSION=11.4
    fi
    if [[ "$BUILD_DIR_EXPLICIT" == 0 ]]; then
        case "$XAVIER_PRESET" in
            jetpack5)
                BUILD_DIR="$PROJECT_DIR/build/local-linux-aarch64-jetpack5-cuda-sm72"
                ;;
            ubuntu20)
                BUILD_DIR="$PROJECT_DIR/build/local-linux-aarch64-ubuntu20-cuda-sm72"
                ;;
            ubuntu22)
                BUILD_DIR="$PROJECT_DIR/build/local-linux-aarch64-ubuntu22-cuda-sm72"
                ;;
        esac
    fi
fi

if [[ -z "$BUILD_DIR" ]]; then
    if [[ "$USE_CUDA" == ON ]]; then
        BUILD_DIR="$PROJECT_DIR/build/local-linux-x64-cuda-sm${CUDA_ARCH}"
    elif [[ "$USE_SYCL" == ON ]]; then
        BUILD_DIR="$PROJECT_DIR/build/local-linux-x64-sycl"
    elif [[ "$USE_VULKAN" == ON ]]; then
        BUILD_DIR="$PROJECT_DIR/build/local-linux-x64-vulkan"
    else
        BUILD_DIR="$PROJECT_DIR/build/local-linux-x64-cpu"
    fi
fi

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
if [[ -n "$CUDA_VERSION" && ! "$CUDA_VERSION" =~ ^[0-9]+([.][0-9]+){1,2}$ ]]; then
    echo "CUDA version must look like 11.4 or 12.4.1" >&2
    exit 2
fi
if [[ "$USE_SYCL" == ON && -n "$SYCL_DEVICE_ARCH" && ! "$SYCL_DEVICE_ARCH" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--sycl-device-arch contains unsupported characters" >&2
    exit 2
fi
if [[ "$USE_SYCL" == ON ]]; then
    command -v icx >/dev/null 2>&1 || {
        echo "--sycl requires Intel oneAPI icx; source oneAPI setvars.sh first" >&2
        exit 1
    }
    command -v icpx >/dev/null 2>&1 || {
        echo "--sycl requires Intel oneAPI icpx; source oneAPI setvars.sh first" >&2
        exit 1
    }
fi
if [[ "$USE_VULKAN" == ON ]]; then
    command -v glslc >/dev/null 2>&1 || {
        echo "--vulkan requires glslc (install the Vulkan shader compiler)" >&2
        exit 1
    }
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
    -DSD_SYCL="$USE_SYCL"
    -DSD_VULKAN="$USE_VULKAN"
    -DGGML_NATIVE=OFF
    -DSDKIT_BUILD_NATIVE_VISION=OFF
    -DSDKIT_BUILD_IMAGE_TOOLS=ON
)
if [[ "$USE_CUDA" == ON ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi
if [[ "$USE_SYCL" == ON ]]; then
    CMAKE_ARGS+=(
        -DCMAKE_C_COMPILER=icx
        -DCMAKE_CXX_COMPILER=icpx
        -DGGML_SYCL=ON
        -DGGML_SYCL_F16=ON
    )
    if [[ -n "$SYCL_DEVICE_ARCH" ]]; then
        CMAKE_ARGS+=(-DGGML_SYCL_DEVICE_ARCH="$SYCL_DEVICE_ARCH")
        echo "Using Intel oneAPI/SYCL backend (AOT architecture $SYCL_DEVICE_ARCH)"
    else
        echo "Using Intel oneAPI/SYCL backend (JIT device code)"
    fi
fi
if [[ "$USE_VULKAN" == ON ]]; then
    echo "Using Vulkan GPU backend"
fi
if [[ -n "$XAVIER_PRESET" ]]; then
    # Xavier is an integrated-memory, single-GPU Jetson platform. Avoid CUDA
    # facilities aimed at desktop/discrete multi-GPU systems. SageAttention is
    # SM80-only and must not be compiled into an SM72 Xavier build.
    CMAKE_ARGS+=(
        -DGGML_CUDA_NCCL=OFF
        -DGGML_CUDA_NO_PEER_COPY=ON
        -DGGML_CUDA_NO_VMM=ON
        -DGGML_CUDA_SAGE=OFF
    )
    CUDA_TOOLKIT_DIR="/usr/local/cuda"
    if [[ -n "$CUDA_VERSION" ]]; then
        CUDA_TOOLKIT_DIR="/usr/local/cuda-$CUDA_VERSION"
    fi
    if [[ -x "$CUDA_TOOLKIT_DIR/bin/nvcc" ]]; then
        CMAKE_ARGS+=(
            -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_DIR/bin/nvcc"
            -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_DIR"
        )
    elif [[ -n "$CUDA_VERSION" ]]; then
        echo "Requested CUDA toolkit not found: $CUDA_TOOLKIT_DIR" >&2
        exit 1
    fi
    echo "Using Xavier preset: $XAVIER_PRESET (AArch64, CUDA ${CUDA_VERSION:-system}, SM72)"
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
