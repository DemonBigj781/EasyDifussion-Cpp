# ROCm / HIP native-call inventory for xFormers

Status: raw inventory. Do not choose normalized translation names from this file yet.

This file records HIP-native primitives that could be used to express the same categories of work as the CUDA xFormers path. Similar names do not automatically imply identical semantics.

## Execution and launch

- `__global__` HIP kernels.
- CUDA-style triple-chevron launch syntax is supported by HIP.
- `hipLaunchKernelGGL(...)` / related HIP launch APIs provide an explicit HIP launch form.
- `dim3` launch geometry.
- `hipStream_t` for ordered asynchronous execution.
- HIP runtime error APIs such as `hipGetLastError()` / return-status handling should be inventoried when the actual ROCm implementation is written.

## Thread/block identity

- HIP exposes CUDA-like built-in execution indices such as `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` in device code.
- `warpSize` must be treated as runtime/target-dependent portable information; AMD documentation explicitly warns not to assume a fixed warp size.

## Synchronization and shared storage

- `__shared__` block/work-group-local storage is available in HIP C++ device code.
- `__syncthreads()` synchronizes threads in a block and provides the corresponding memory visibility semantics.
- HIP also exposes `__syncthreads_count`, `__syncthreads_and`, and `__syncthreads_or`.
- HIP provides warp cross-lane/shuffle operations including `__shfl`, `__shfl_up`, `__shfl_down`, and `__shfl_xor`, plus sync variants on current ROCm.
- HIP warp masks are 64-bit even on targets where the active warp width is smaller, which is a portability difference that must be preserved for later comparison.

## Memory ordering / fences

- `__threadfence_block()`
- `__threadfence()`
- `__threadfence_system()`

These are relevant if a later HIP implementation requires ordering beyond the implicit fence behavior of a block barrier.

## Atomics

HIP exposes native atomics including:

- `atomicAdd`
- `atomicSub`
- `atomicMin`
- `atomicMax`
- `atomicExch`
- `atomicCAS`
- bitwise atomics

HIP also distinguishes safe/unsafe floating-point atomic paths on some AMD hardware. Current CUDA xFormers code does not require atomics, but they are recorded because a different architecture-specific implementation may.

## Scalar and dtype considerations

HIP supports half/bfloat-style GPU types and conversion intrinsics through ROCm headers, but the exact types/conversions to use for this project must be recorded from the implementation selected for each AMD architecture rather than assumed from CUDA spelling.

## Architecture-specific facts to preserve

- AMD wavefront/warp width may differ by architecture; portable code must not hard-code NVIDIA's width.
- HIP's CUDA-like source syntax does not guarantee identical hardware behavior.
- Some cross-lane mask conventions differ from CUDA, including 64-bit masks in HIP.
- Hardware support and numerical behavior of some floating-point atomics varies by AMD architecture.
- HIP does not support CUDA dynamic parallelism; not currently needed by xFormers, but this is an example of why source-level similarity must not be mistaken for complete execution equivalence.

## Not yet claimed

This document does **not** claim that the current CUDA xFormers kernel can be mechanically hipified and considered correct. The actual ROCm implementation may need a different reduction strategy, wave-aware behavior, layout handling, or fused procedure.
