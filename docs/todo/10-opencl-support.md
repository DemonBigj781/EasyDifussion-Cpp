# 10 — OpenCL Support

## Objective
Provide a broad compatibility GPU-compute path for hardware that lacks a better supported native backend.

## Implementation
1. Check current GGML/stable-diffusion.cpp OpenCL support and reuse it if suitable.
2. Enumerate platforms/devices and expose stable device selection plus memory/capability information.
3. Implement or enable buffer allocation, transfer queues, synchronization, and kernel compilation/cache.
4. Start with the core operations required by a minimal model path: matmul, elementwise operations, normalization, activations, and required attention primitives.
5. Add dtype/extension capability checks; do not assume FP16 or subgroup extensions exist on all devices.
6. Avoid vendor-specific extensions unless guarded by capability tests and generic fallbacks.
7. Integrate with the same model placement/offload policy as other backends where feasible.

## Role
Treat OpenCL primarily as a compatibility backend. CUDA, ROCm, and oneAPI/SYCL should remain preferred when they provide better feature coverage/performance.

## Validation
Run kernel unit tests on at least two vendors if possible, then a small end-to-end model. Test missing-FP16 devices, OOM, cancellation, and repeated context creation.

## Complete when
A documented set of OpenCL devices can run either the full supported pipeline or a clearly-defined subset without hidden backend assumptions.
