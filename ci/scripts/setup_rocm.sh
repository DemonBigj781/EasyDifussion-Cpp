#!/usr/bin/env bash
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.2.2}"
UBUNTU_CODENAME="${UBUNTU_CODENAME:-noble}"

sudo mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
  | gpg --dearmor \
  | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} ${UBUNTU_CODENAME} main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null

sudo tee /etc/apt/preferences.d/rocm-pin >/dev/null <<'EOF'
Package: rocm-* hip* roc* hsa-* amd-smi* comgr* miopen* rccl* llvm-amdgpu* openmp-extras*
Pin: origin repo.radeon.com
Pin-Priority: 1001
EOF

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  rocm-dev \
  hipblas-dev \
  rocblas-dev

apt-cache policy hipcc rocm-cmake rocm-dev | sed -n '1,120p'
