# 15 — DirectML / D3D12 Backend

## Objective
Add a native Windows DirectML / Direct3D 12 compute backend to EasyDifussion-Cpp, with a focus on Windows GPU memory behavior and compatibility across NVIDIA, AMD, and Intel hardware through D3D12.

## Implementation
1. Add a project-level backend option for DirectML / D3D12 and ensure it is mutually exclusive with CUDA, ROCm/HIP, Vulkan, OpenCL, OpenGL, SYCL/oneAPI, Metal, and MUSA when selected.
2. Add Windows build integration for the Direct3D 12 and DirectML SDK/runtime interfaces using the Windows SDK and DirectML headers/libraries.
3. Implement D3D12 device and adapter enumeration, preferring a requested adapter while allowing automatic selection.
4. Implement GPU buffer allocation, upload/readback buffers, descriptor heaps, command queues, command lists, fences, and synchronization.
5. Implement DirectML tensor descriptors and operator dispatch for the operations required by stable-diffusion.cpp / GGML.
6. Add capability checks so unsupported operators fall back cleanly rather than producing invalid GPU work.
7. Investigate D3D12 residency, pageable resources, shared GPU memory, budget queries, and controlled VRAM-to-system-memory pressure behavior. Do not assume oversubscription semantics without validating them on real hardware.
8. Expose memory budget, current usage, dedicated VRAM, shared system memory, and residency state to the backend diagnostics.
9. Keep the DirectML / D3D12 backend isolated from CUDA/ROCm/oneAPI toolchain installation so selecting this backend does not install or compile unrelated accelerator stacks.
10. Once the stable-diffusion.cpp implementation is stable, port the same backend contract into the isolated llama.cpp GGML tree rather than creating a separate incompatible implementation.

## GitHub Actions
- Add `directml-d3d12` to the manual Windows Compile backend selector.
- Add `directml-d3d12` to the manual Windows Regression backend selector.
- Until the backend implementation is compilable, selecting it must fail explicitly with an implementation-not-wired message rather than silently compiling CPU or another backend.
- When implementation begins compiling, Windows Compile should build only DirectML / D3D12 and verify the output artifact without running inference.
- Windows Regression should perform configuration/source validation without running generated binaries unless a later dedicated runtime test is intentionally added.

## Dependencies
- Windows SDK with Direct3D 12 headers/libraries.
- DirectML headers and runtime package/library.
- A clear GGML backend abstraction for buffers, devices, operator support, and graph dispatch.

## Validation
Validate configuration and compilation first. Hardware validation should later cover NVIDIA, AMD, and Intel adapters; dedicated-VRAM-only workloads; workloads that exceed dedicated VRAM; memory budget changes; repeated load/unload; cancellation; and fallback behavior for unsupported operators.

## Complete when
The DirectML / D3D12 backend can be selected independently, compiles without enabling other accelerator APIs, exposes a functional GGML/stable-diffusion compute device, handles supported graphs correctly, falls back safely for unsupported operations, and has documented/validated GPU-memory residency behavior. Llama integration may follow as a separate promotion step once the backend contract is stable.
