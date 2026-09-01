FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ROCM_VERSION=7.2.2
ARG UBUNTU_CODENAME=noble

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Base build and CI tooling plus portable graphics/compute API development headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    ca-certificates \
    curl \
    wget \
    gnupg \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    ocl-icd-opencl-dev \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    libvulkan-dev \
    vulkan-tools \
    glslc \
    glslang-tools \
    spirv-headers \
    spirv-tools \
    mesa-common-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libdrm-dev \
    && rm -rf /var/lib/apt/lists/*

# oneAPI / SYCL / Level Zero: compiler plus libraries used by CPU/GPU/NPU-facing code.
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main' \
      > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-dpcpp-cpp \
      intel-oneapi-mkl-devel \
      intel-oneapi-dnnl-devel \
      intel-ocloc \
      libze1 \
      libze-dev \
    && rm -rf /var/lib/apt/lists/*

# OpenVINO official APT distribution. This supplies the C/C++ runtime API and headers.
RUN echo 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/openvino ubuntu24 main' \
      > /etc/apt/sources.list.d/intel-openvino.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends openvino \
    && rm -rf /var/lib/apt/lists/*

# ROCm/HIP development stack. Keep AMD packages preferred over Ubuntu's older
# matching package names, mirroring ci/scripts/setup_rocm.sh without sudo.
RUN mkdir -p /etc/apt/keyrings \
    && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor > /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} ${UBUNTU_CODENAME} main" \
      > /etc/apt/sources.list.d/rocm.list \
    && printf '%s\n' \
      'Package: rocm-* hip* roc* hsa-* amd-smi* comgr* miopen* rccl* llvm-amdgpu* openmp-extras*' \
      'Pin: origin repo.radeon.com' \
      'Pin-Priority: 1001' \
      > /etc/apt/preferences.d/rocm-pin \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      rocm-dev \
      hipblas-dev \
      rocblas-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/rocm/bin:/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV ROCM_PATH="/opt/rocm"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Make oneAPI available in interactive shells while workflows can still source
# setvars.sh explicitly for the complete environment.
RUN echo 'if [ -f /opt/intel/oneapi/setvars.sh ]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; fi' \
      > /etc/profile.d/oneapi.sh

# Fail the image build if any API family expected by 990 is absent, and print
# the exact component that failed so CI logs are actionable.
RUN set -eux; \
    source /opt/intel/oneapi/setvars.sh >/dev/null; \
    check_file() { label="$1"; path="$2"; echo "[990 verify] $label -> $path"; test -f "$path" || { echo "[990 missing] $label: $path" >&2; exit 1; }; }; \
    check_exec() { label="$1"; path="$2"; echo "[990 verify] $label -> $path"; test -x "$path" || { echo "[990 missing] $label: $path" >&2; exit 1; }; }; \
    check_cmd() { label="$1"; cmd="$2"; echo "[990 verify] $label -> $cmd"; command -v "$cmd" || { echo "[990 missing] $label command: $cmd" >&2; exit 1; }; }; \
    check_find() { label="$1"; pattern="$2"; echo "[990 verify] $label -> $pattern"; found="$(find /usr /opt/intel -path "$pattern" -print -quit 2>/dev/null)"; test -n "$found" || { echo "[990 missing] $label pattern: $pattern" >&2; exit 1; }; echo "[990 found] $label -> $found"; }; \
    check_exec CUDA /usr/local/cuda/bin/nvcc; \
    check_exec ROCm-HIP /opt/rocm/bin/hipcc; \
    check_file OpenCL /usr/include/CL/cl.h; \
    check_file Vulkan /usr/include/vulkan/vulkan.h; \
    check_file OpenGL /usr/include/GL/gl.h; \
    check_file EGL /usr/include/EGL/egl.h; \
    check_find Level-Zero '*/level_zero/ze_api.h'; \
    check_find OpenVINO '*/openvino/openvino.hpp'; \
    check_cmd GLSLC glslc; \
    check_cmd GLSLang glslangValidator; \
    check_cmd SPIRV-Tools spirv-val; \
    check_cmd oneAPI-ICPX icpx; \
    echo '[990 verify] universal toolchain validation complete'

CMD ["/bin/bash"]
