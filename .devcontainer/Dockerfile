FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ROCM_VERSION=7.2.2
ARG UBUNTU_CODENAME=noble

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 990 Phase 00: base build / CPU development tools.
RUN echo '[990 phase 00] base build and CPU tools' \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      clang \
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
      libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# 990 Phase 10: CUDA. The CUDA development toolkit comes from the base image.
RUN echo '[990 phase 10] CUDA' \
    && test -x /usr/local/cuda/bin/nvcc \
    && /usr/local/cuda/bin/nvcc --version

# 990 Phase 20: ROCm / HIP.
RUN echo '[990 phase 20] ROCm / HIP' \
    && mkdir -p /etc/apt/keyrings \
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

# 990 Phase 30: oneAPI / SYCL / oneDNN / oneMKL / Level Zero.
RUN echo '[990 phase 30] oneAPI / SYCL / Level Zero' \
    && wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
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

# 990 Phase 50: OpenCL.
RUN echo '[990 phase 50] OpenCL' \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ocl-icd-opencl-dev \
      ocl-icd-libopencl1 \
      opencl-headers \
      clinfo \
    && rm -rf /var/lib/apt/lists/*

# 990 Phase 60: OpenVINO C/C++ runtime and headers.
RUN echo '[990 phase 60] OpenVINO' \
    && echo 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/openvino ubuntu24 main' \
      > /etc/apt/sources.list.d/intel-openvino.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends openvino \
    && rm -rf /var/lib/apt/lists/*

# 990 Phase 100: Vulkan / SPIR-V.
RUN echo '[990 phase 100] Vulkan / SPIR-V' \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      libvulkan-dev \
      vulkan-tools \
      glslc \
      glslang-tools \
      spirv-headers \
      spirv-tools \
    && rm -rf /var/lib/apt/lists/*

# 990 Phase 110/130: OpenGL / EGL / Mesa.
RUN echo '[990 phase 110/130] OpenGL / EGL / Mesa' \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      mesa-common-dev \
      libgl1-mesa-dev \
      libegl1-mesa-dev \
      libgles2-mesa-dev \
      libdrm-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/rocm/bin:/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV ROCM_PATH="/opt/rocm"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN echo 'if [ -f /opt/intel/oneapi/setvars.sh ]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; fi' \
      > /etc/profile.d/oneapi.sh

# 990 final validation: report every missing component in one run, then fail once.
RUN set -u; \
    failures=0; \
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || { echo '[990 missing] oneAPI setvars.sh could not be sourced' >&2; failures=$((failures+1)); }; \
    check_file() { label="$1"; path="$2"; if test -f "$path"; then echo "[990 ok] $label -> $path"; else echo "[990 missing] $label -> $path" >&2; failures=$((failures+1)); fi; }; \
    check_exec() { label="$1"; path="$2"; if test -x "$path"; then echo "[990 ok] $label -> $path"; else echo "[990 missing] $label -> $path" >&2; failures=$((failures+1)); fi; }; \
    check_cmd() { label="$1"; cmd="$2"; if path="$(command -v "$cmd" 2>/dev/null)"; then echo "[990 ok] $label -> $path"; else echo "[990 missing] $label command -> $cmd" >&2; failures=$((failures+1)); fi; }; \
    check_find() { label="$1"; pattern="$2"; found="$(find /usr /opt/intel -path "$pattern" -print -quit 2>/dev/null)"; if test -n "$found"; then echo "[990 ok] $label -> $found"; else echo "[990 missing] $label pattern -> $pattern" >&2; failures=$((failures+1)); fi; }; \
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
    if test "$failures" -ne 0; then echo "[990 verify] FAILED: $failures component(s) missing" >&2; exit 1; fi; \
    echo '[990 verify] universal toolchain validation complete'

CMD ["/bin/bash"]
