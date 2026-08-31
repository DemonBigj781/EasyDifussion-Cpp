# API.cpp

`source/API.cpp` is the common compute API tree used to organize backend-specific implementations behind one project-facing interface.

Layout:

```text
source/API.cpp/
├── common/      # Backend-neutral contracts, types, capability checks, and dispatch
├── cpu/         # CPU implementations
├── cuda/        # NVIDIA CUDA implementations
├── rocm/        # AMD ROCm/HIP implementations
├── oneapi/      # Intel oneAPI/SYCL implementations
└── vulkan/      # Vulkan compute implementations
```

The intended form is `API.cpp/<backend>/<code>`.

Backend-specific code stays inside its backend directory. Code outside `API.cpp` should call the common API instead of directly depending on a backend implementation when a common operation exists.

The stable-diffusion attention layer is owned here: backend-neutral FlexAttention and the common attention contract live under `common/attention`, while CUDA FlashAttention, SageAttention, xFormers-compatible forward attention, and their template instantiations live under `cuda/attention`. The legacy stable-diffusion.cpp paths contain compatibility forwarding files only. Shared ggml CUDA primitives that are also used by non-attention operations remain in ggml and are reached through `cuda/common.cuh`.

The dedicated `99 - API` workflow is intended to compile/test this API layer in isolation. Normal CUDA, HIP, and MUSA builds compile their corresponding canonical `API.cpp` attention sources directly.
