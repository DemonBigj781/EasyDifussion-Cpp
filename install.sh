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
  --rocm          Build for AMD GPUs with ROCm/HIP.
  --rocm-target ARCH
                  AMD GPU architecture to compile for (default: gfx1030).
  --sycl          Build for Intel GPUs with oneAPI/SYCL (source setvars.sh first).
  --oneapi-cpu    Build for Intel Xeon/CPU with Intel LLVM + oneMKL BLAS.
  --sycl-device ARCH
                  Optional Intel SYCL AOT device architecture (spir64_gen).
  --sycl-no-dnn   Disable oneDNN acceleration in the GGML SYCL backend.
  --sycl-no-level-zero
                  Disable direct Level Zero support in the GGML SYCL backend.
  --cpu           Build llama.cpp without CUDA/SYCL/ROCm/oneAPI CPU tuning.
  --jobs N        Set the parallel build job count.
  -h, --help      Show this help.

ROCm mode expects a working ROCm development toolchain with hipcc, clang,
hipBLAS, and rocBLAS available. The selected --rocm-target is propagated to
both the diffusion GGML backend and the isolated llama.cpp runtime so the
native bundle is compiled consistently for the same AMD architecture.

For oneAPI GPU builds, source Intel's setvars.sh first. For oneAPI CPU builds,
setvars.sh must also expose icx, icpx, and MKLROOT. The oneAPI CPU mode uses
Intel LLVM for compilation, oneMKL through GGML's BLAS backend, and GGML native
CPU tuning so Xeon ISA features can be selected by the compiler/runtime.
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
ROCM_TARGET="${EASY_DIFFUSION_ROCM_TARGET:-gfx1030}"
SYCL_DEVICE_ARCH="${EASY_DIFFUSION_SYCL_DEVICE_ARCH:-}"
SYCL_DNN=ON
SYCL_LEVEL_ZERO=ON
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
        --rocm)
            CUDA_MODE=rocm
            ;;
        --rocm-target)
            shift
            [ "$#" -gt 0 ] || fail "--rocm-target requires an AMD GPU architecture such as gfx1030"
            ROCM_TARGET="$1"
            ;;
        --sycl)
            CUDA_MODE=sycl
            ;;
        --oneapi-cpu)
            CUDA_MODE=oneapi-cpu
            ;;
        --sycl-device)
            shift
            [ "$#" -gt 0 ] || fail "--sycl-device requires an Intel device architecture"
            SYCL_DEVICE_ARCH="$1"
            ;;
        --sycl-no-dnn)
            SYCL_DNN=OFF
            ;;
        --sycl-no-level-zero)
            SYCL_LEVEL_ZERO=OFF
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
HIP_ENABLED=OFF
SYCL_ENABLED=OFF
BUILD_PLATFORM=cpu
COMPILER_ARGS=()
SYCL_CMAKE_ARGS=()
CPU_CMAKE_ARGS=()
HIP_CMAKE_ARGS=()

if [ "$CUDA_MODE" = cuda ]; then
    command -v nvcc >/dev/null || fail "--cuda was requested but nvcc is unavailable"
    command -v nvidia-smi >/dev/null || fail "--cuda was requested but nvidia-smi is unavailable"
    CUDA_ENABLED=ON
    BUILD_PLATFORM=cuda
elif [ "$CUDA_MODE" = rocm ]; then
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export PATH="$ROCM_PATH/bin:$PATH"
    command -v hipcc >/dev/null || fail "--rocm requires hipcc from a ROCm development installation"
    [ -x "$ROCM_PATH/llvm/bin/clang" ] || fail "--rocm requires ROCm clang at $ROCM_PATH/llvm/bin/clang"
    [ -x "$ROCM_PATH/llvm/bin/clang++" ] || fail "--rocm requires ROCm clang++ at $ROCM_PATH/llvm/bin/clang++"
    HIP_ENABLED=ON
    BUILD_PLATFORM=rocm
    COMPILER_ARGS=(
        "-DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang"
        "-DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++"
    )
    HIP_CMAKE_ARGS=(
        "-DGPU_TARGETS=$ROCM_TARGET"
        "-DAMDGPU_TARGETS=$ROCM_TARGET"
        "-DCMAKE_HIP_ARCHITECTURES=$ROCM_TARGET"
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
    echo "ROCm/HIP mode: target=$ROCM_TARGET rocm_path=$ROCM_PATH"
elif [ "$CUDA_MODE" = sycl ]; then
    command -v icx >/dev/null || fail "--sycl requires the oneAPI icx compiler (source setvars.sh first)"
    command -v icpx >/dev/null || fail "--sycl requires the oneAPI icpx compiler (source setvars.sh first)"
    command -v sycl-ls >/dev/null || fail "--sycl requires the oneAPI SYCL runtime"
    SYCL_GPU_LINES="$(sycl-ls 2>/dev/null | grep -E '\[(level_zero|opencl):gpu\]' || true)"
    [ -n "$SYCL_GPU_LINES" ] || fail "oneAPI does not currently expose an Intel GPU; install/verify the card driver first"
    echo "Detected oneAPI GPU devices:"
    printf '%s\n' "$SYCL_GPU_LINES"
    SYCL_ENABLED=ON
    BUILD_PLATFORM=sycl
    COMPILER_ARGS=(-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
    SYCL_CMAKE_ARGS=(
        -DGGML_SYCL_TARGET=INTEL
        -DGGML_SYCL_F16=ON
        -DGGML_SYCL_DNN="$SYCL_DNN"
        -DGGML_SYCL_SUPPORT_LEVEL_ZERO="$SYCL_LEVEL_ZERO"
    )
    if [ -n "$SYCL_DEVICE_ARCH" ]; then
        SYCL_CMAKE_ARGS+=("-DGGML_SYCL_DEVICE_ARCH=$SYCL_DEVICE_ARCH")
        echo "SYCL AOT device architecture: $SYCL_DEVICE_ARCH"
    else
        echo "SYCL device code mode: JIT (portable Intel target)"
    fi
elif [ "$CUDA_MODE" = oneapi-cpu ]; then
    command -v icx >/dev/null || fail "--oneapi-cpu requires Intel oneAPI icx (source setvars.sh first)"
    command -v icpx >/dev/null || fail "--oneapi-cpu requires Intel oneAPI icpx (source setvars.sh first)"
    [ -n "${MKLROOT:-}" ] || fail "--oneapi-cpu requires MKLROOT from oneAPI setvars.sh"
    BUILD_PLATFORM=oneapi-cpu
    COMPILER_ARGS=(-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
    CPU_CMAKE_ARGS=(
        -DGGML_BLAS=ON
        -DGGML_BLAS_VENDOR=Intel10_64lp_seq
        -DGGML_NATIVE=ON
    )
    echo "oneAPI CPU mode: Intel LLVM + oneMKL + native Xeon ISA tuning"
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
        "${SYCL_CMAKE_ARGS[@]}" \
        "${CPU_CMAKE_ARGS[@]}" \
        "${HIP_CMAKE_ARGS[@]}" \
        -DGGML_CUDA="$CUDA_ENABLED" \
        -DGGML_HIP="$HIP_ENABLED" \
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
    elif [ "$BUILD_PLATFORM" = rocm ]; then
        SDKIT_VARIANT="$ROCM_TARGET"
    elif [ "$BUILD_PLATFORM" = sycl ]; then
        if [ -n "$SYCL_DEVICE_ARCH" ]; then
            SDKIT_VARIANT="intel-$SYCL_DEVICE_ARCH"
        else
            SDKIT_VARIANT="intel-jit"
        fi
    elif [ "$BUILD_PLATFORM" = oneapi-cpu ]; then
        SDKIT_VARIANT="xeon-mkl-native"
    fi
    SDKIT_TARGET_DIR="$PROJECT_ROOT/backends/sdkit3/linux-x64-$BUILD_PLATFORM-$SDKIT_VARIANT"
    echo "Configuring sdkit/stable-diffusion.cpp ($BUILD_PLATFORM) with the integrated llama.cpp runtime..."
    cmake -S "$SDKIT_SOURCE" -B "$SDKIT_BUILD_DIR" "${GENERATOR_ARGS[@]}" \
        "${COMPILER_ARGS[@]}" \
        "${SYCL_CMAKE_ARGS[@]}" \
        "${CPU_CMAKE_ARGS[@]}" \
        "${HIP_CMAKE_ARGS[@]}" \
        -DSD_CUDA="$CUDA_ENABLED" \
        -DSD_HIPBLAS="$HIP_ENABLED" \
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
