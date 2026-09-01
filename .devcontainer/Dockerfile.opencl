ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_VERSION=24.04
ARG OPENCL_VERSION=3.0
ARG OPENCL_TARGET_VERSION=300
ARG OPENCL_RUNTIME_PROFILE=full
ARG PYTHON_VERSION=none
ARG PYTORCH_VERSION=none
ARG PYTORCH_INDEX_URL=none

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="EasyDifussion-Cpp 040 OpenCL Toolchain" \
      io.easydifussion.opencl.version="${OPENCL_VERSION}" \
      io.easydifussion.opencl.runtime-profile="${OPENCL_RUNTIME_PROFILE}" \
      io.easydifussion.ubuntu.version="${UBUNTU_VERSION}" \
      io.easydifussion.python.version="${PYTHON_VERSION}" \
      io.easydifussion.pytorch.version="${PYTORCH_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ clang cmake ninja-build pkg-config git ca-certificates curl wget \
    make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncurses-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Required OpenCL SDK surface. Keep these separate from optional runtimes so APT failures are obvious.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends opencl-headers; \
    apt-get install -y --no-install-recommends ocl-icd-opencl-dev; \
    if apt-cache policy clinfo 2>/dev/null | grep -q 'Candidate: [^()]'; then \
      apt-get install -y --no-install-recommends clinfo || echo '[040 warning] clinfo install failed; continuing because it is diagnostic-only'; \
    else \
      echo '[040 warning] clinfo has no install candidate on this Ubuntu release'; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Optional OpenCL runtime implementations. Missing distro packages do not invalidate the SDK image.
RUN set -eux; \
    apt-get update; \
    install_optional() { \
      p="$1"; \
      candidate="$(apt-cache policy "$p" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"; \
      if [ -z "$candidate" ] || [ "$candidate" = '(none)' ]; then \
        echo "[040 optional] $p unavailable on Ubuntu ${UBUNTU_VERSION}; skipping"; \
      elif apt-get install -y --no-install-recommends "$p"; then \
        echo "[040 optional] installed $p ($candidate)"; \
      else \
        echo "[040 warning] $p exists but could not be installed; skipping optional runtime package"; \
        apt-get -f install -y || true; \
      fi; \
    }; \
    case "$OPENCL_RUNTIME_PROFILE" in \
      headers-only) echo '[040] headers-only profile selected' ;; \
      pocl) install_optional pocl-opencl-icd; install_optional libpocl-dev ;; \
      mesa) install_optional mesa-opencl-icd; install_optional libclc-dev ;; \
      full) install_optional pocl-opencl-icd; install_optional libpocl-dev; install_optional mesa-opencl-icd; install_optional libclc-dev ;; \
      *) echo "Unknown OpenCL runtime profile: $OPENCL_RUNTIME_PROFILE" >&2; exit 1 ;; \
    esac; \
    rm -rf /var/lib/apt/lists/*

ENV CL_TARGET_OPENCL_VERSION="${OPENCL_TARGET_VERSION}"
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"

RUN git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT" \
    && if [ "$PYTHON_VERSION" != "none" ]; then \
         pyenv install "$PYTHON_VERSION" \
         && pyenv global "$PYTHON_VERSION" \
         && python -m ensurepip --upgrade \
         && python -m pip install --upgrade pip setuptools wheel; \
       fi

RUN if [ "$PYTORCH_VERSION" != "none" ]; then \
      if [ "$PYTHON_VERSION" = "none" ]; then echo 'PyTorch requires Python' >&2; exit 1; fi; \
      if [ "$PYTORCH_INDEX_URL" = "none" ]; then \
        python -m pip install "torch==${PYTORCH_VERSION}"; \
      else \
        python -m pip install "torch==${PYTORCH_VERSION}" --index-url "$PYTORCH_INDEX_URL"; \
      fi; \
    fi

RUN printf '%s\n' \
    'export PYENV_ROOT="$HOME/.pyenv"' \
    'export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"' \
    'if command -v pyenv >/dev/null 2>&1; then eval "$(pyenv init -)"; fi' \
    > /etc/profile.d/pyenv.sh

RUN set -eux; \
    test -f /usr/include/CL/cl.h; \
    test -f /usr/include/CL/cl_platform.h; \
    printf '%s\n' \
      "#define CL_TARGET_OPENCL_VERSION ${OPENCL_TARGET_VERSION}" \
      '#include <CL/cl.h>' \
      'int main(void) {' \
      '    cl_uint n = 0;' \
      '    cl_int rc = clGetPlatformIDs(0, 0, &n);' \
      '    return (rc == CL_SUCCESS || rc == CL_PLATFORM_NOT_FOUND_KHR) ? 0 : 1;' \
      '}' \
      > /tmp/opencl-check.c; \
    gcc /tmp/opencl-check.c -lOpenCL -o /tmp/opencl-check; \
    /tmp/opencl-check; \
    rm -f /tmp/opencl-check /tmp/opencl-check.c; \
    if [ "$PYTHON_VERSION" != "none" ]; then python --version; fi; \
    if [ "$PYTORCH_VERSION" != "none" ]; then python -c 'import torch; print(torch.__version__)'; fi; \
    echo "[040 verify] OpenCL ${OPENCL_VERSION} / ${OPENCL_RUNTIME_PROFILE} / Ubuntu ${UBUNTU_VERSION} validation complete"

CMD ["/bin/bash"]
