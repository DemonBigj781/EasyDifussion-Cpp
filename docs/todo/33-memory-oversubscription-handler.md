# 33 — Memory Oversubscription Handler

## Objective
Add a backend-aware memory oversubscription handler that detects VRAM pressure before allocation failure and can move, stage, or recreate eligible data in system RAM when a model or workload exceeds available device memory.

## Scope
The handler should be usable by Stable Diffusion, the isolated llama runtime, and future backends without hard-coding one API. Backend-specific implementations may use different mechanisms, but they should report through one common policy and telemetry layer.

## Implementation
1. Add a central memory-budget interface that reports dedicated device memory, currently committed device memory, reclaimable allocations, host RAM availability, and backend-specific shared/unified memory capabilities.
2. Track allocations by class: model weights, KV/cache-like persistent state, temporary tensors, activations, decoded/encoded intermediates, and disposable caches.
3. Introduce configurable watermarks instead of waiting for an out-of-memory error. Suggested states are normal, pressure, critical, and recovery.
4. Under pressure, first release disposable caches and temporary allocations that can be reconstructed cheaply.
5. If pressure remains, migrate or stage eligible tensors to host RAM using the selected backend's supported mechanism. Do not assume every backend can transparently oversubscribe VRAM.
6. When a backend supports pageable/shared/unified memory, expose that capability through the common handler and prefer the native mechanism over manual copies.
7. For DirectML/D3D12, investigate residency management, shared GPU memory, pageable device-local memory behavior, and explicit resource eviction/make-resident policy.
8. For CUDA, distinguish native Linux HMM/Unified Memory, explicit cudaMallocManaged-style paths where appropriate, ordinary host offload, and GPUs/drivers where oversubscription is not viable.
9. For ROCm/HIP, investigate HMM/XNACK-capable unified memory and explicit host staging, with conservative fallback when unavailable.
10. For oneAPI/SYCL, investigate USM shared allocations, device/host USM migration behavior, and Level Zero memory properties.
11. For Vulkan, OpenGL, and OpenCL, implement explicit staging/offload policies unless a proven backend-specific shared-memory mechanism exists.
12. Add hysteresis so allocations are not continuously moved between VRAM and RAM around a single threshold.
13. Add an allocation retry path: reclaim, offload/migrate, retry once under the handler, then return a clear OOM rather than looping indefinitely.
14. Preserve correctness during cancellation, model unload, backend reset, and process shutdown. Every migrated allocation must have one owner and one valid source of truth.
15. Record telemetry for pressure events, bytes reclaimed, bytes migrated, migration latency, retries, backend mechanism used, and whether recovery succeeded.
16. Add user-facing policy settings such as disabled, conservative, balanced, aggressive, plus optional reserve-VRAM and max-host-RAM limits.
17. Ensure the handler can be disabled per backend if testing shows that oversubscription is slower or less reliable than an immediate controlled fallback.

## Safety / Correctness Rules
- Never silently reinterpret an allocation as shared/unified memory unless the backend guarantees the semantics required by that tensor.
- Never evict data that cannot be reconstructed or restored before its next use.
- Do not allow host-memory oversubscription to push the operating system into uncontrolled swap thrashing without a configured limit.
- Treat multi-GPU memory budgets independently unless a backend explicitly supports peer-access residency for the allocation.
- Prefer predictable degraded performance over process termination or corrupted tensors.

## Validation
- Synthetic allocation tests that deliberately cross 70%, 85%, 95%, and 100% of available VRAM.
- Stable Diffusion model load and generation with workloads slightly below, at, and above physical VRAM capacity.
- Repeated load/unload and cancellation cycles to detect leaked VRAM or RAM.
- Multi-GPU tests where only one device is pressured.
- Backend-specific tests for DirectML/D3D12, CUDA, ROCm, oneAPI/SYCL, Vulkan, OpenGL, and OpenCL as each implementation becomes available.
- Confirm that disabling oversubscription restores the existing allocation behavior exactly.
- Verify that a failed recovery produces a bounded, explicit OOM instead of a hang or infinite retry loop.

## Complete when
The project can detect impending device-memory exhaustion, reclaim or move eligible allocations according to a documented backend policy, continue workloads beyond the normal dedicated-VRAM limit where the backend supports it, and fail cleanly where it does not, without leaks, corruption, uncontrolled host-memory growth, or infinite allocation retries.
