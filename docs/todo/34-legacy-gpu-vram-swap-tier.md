# 34 — Legacy GPU VRAM Swap / Memory Tier

## Objective
Allow otherwise unused or inference-unsupported GPUs to contribute their VRAM as an optional memory spill tier instead of leaving that memory idle. This is not literally Linux zram; it is a VRAM-backed swap/cache tier that may sit alongside RAM, zram, and disk swap.

## Implementation
1. Enumerate secondary GPUs and accelerators separately from devices selected for inference.
2. Determine whether each unused device still exposes a safe VRAM allocation/copy mechanism through an available driver or API even when its architecture is unsupported by the active inference backend.
3. Never use a device that lacks a working OS/driver memory-access path. “Unsupported for inference” is acceptable; “unreachable by the driver” is not.
4. Add a `legacy_vram_tier` policy with explicit `off`, `auto`, and device-selection modes. Default must remain `off` until the feature has been validated.
5. Implement a backend-neutral VRAM-tier interface supporting allocate, free, copy-in, copy-out, capacity, pressure, health, and teardown operations.
6. Provide adapters where practical for CUDA driver allocations, HIP/ROCm, OpenCL, oneAPI/SYCL, Vulkan memory, DirectML/D3D12 heaps, or another safe platform-specific mechanism.
7. On Linux, investigate exposing reserved VRAM as a block/swap-like tier where that is safer than application-managed paging. Keep kernel/NBD-style mechanisms optional and isolated from the normal inference process.
8. Integrate the tier with the memory oversubscription handler. Preferred policy should be configurable, for example active GPU VRAM -> spare GPU VRAM -> host RAM/zram -> disk, or active GPU VRAM -> host RAM -> spare GPU VRAM depending on measured PCIe and compression costs.
9. Treat VRAM belonging to a display GPU conservatively. Reserve enough memory for the desktop/display stack and do not claim all reported free VRAM.
10. Do not allow the inference backend and VRAM-tier allocator to independently own the same physical device unless an explicit coexistence policy and capacity reservation are active.
11. Support multiple spare GPUs as independent tiers with capacity, bandwidth, PCIe topology, NUMA location, and health metadata.
12. Benchmark transfer bandwidth and latency before automatically enabling a device. A very slow PCIe link may be worse than compressed host RAM.
13. Add failure containment: if a spare GPU resets, disappears, or reports a copy/allocation failure, stop placing new pages there and recover or fail affected allocations without corrupting model state.
14. Make all tier usage observable: device identity, bytes reserved, bytes occupied, bytes moved in/out, transfer rate, eviction count, failures, and fallback destination.
15. Ensure clean shutdown releases all GPU allocations and any optional block/NBD resources before driver unload or process exit.

## Safety / correctness rules
- Never overwrite framebuffer/display-critical allocations.
- Never assume VRAM is persistent storage.
- Never use undocumented physical BAR addresses as a default implementation path.
- Never advertise a legacy GPU as a compute backend merely because its VRAM can be accessed.
- Never silently enable system swap on a GPU without an explicit configuration setting.
- Keep a guaranteed fallback path to ordinary host RAM/offload when the VRAM tier is unavailable.

## Relationship to memory oversubscription
TODO 33 owns global pressure policy and migration decisions. This objective provides an additional storage tier that TODO 33 may select. The oversubscription handler must remain functional when no spare GPU exists.

## Validation
Test with one and multiple unused GPUs, unsupported-for-inference but driver-visible GPUs, display and headless GPUs, constrained PCIe links, near-full VRAM, repeated allocation/free cycles, model load/unload, forced tier exhaustion, GPU reset/removal where safely testable, host-memory pressure, and fallback to RAM/zram/disk.

Compare latency and throughput against host RAM, compressed RAM/zram, and disk swap. Confirm that enabling the feature never changes which GPU performs inference unless separately configured.

## Complete when
An otherwise unused but driver-accessible GPU can safely provide a bounded VRAM memory tier, the oversubscription handler can spill to and recover from that tier, failures fall back cleanly, display/inference allocations remain protected, and disabling the feature restores normal behavior with no residual GPU or block-device resources.
