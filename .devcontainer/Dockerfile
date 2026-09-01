FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ROCM_VERSION=7.2.2
ARG UBUNTU_CODENAME=noble

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Base build and CI tooling.
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
    && rm -rf /var/lib/apt/lists/*

# oneAPI: compiler plus the libraries used by the backend workflows.
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

CMD ["/bin/bash"]
