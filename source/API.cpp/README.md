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

Attention implementations such as Flash Attention compatibility, Sage Attention, and xFormers-compatible paths can be migrated into this tree incrementally. Existing working implementations should not be moved until their replacement path is ready and individually build-tested.

The dedicated `99 - API` workflow is intended to compile/test this API layer in isolation. Normal backend workflows should eventually compile their corresponding `API.cpp/<backend>` implementation as part of the full project build.
