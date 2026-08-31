# Legacy GPU VRAM Memory Tier

## Overview

EasyDifussion-Cpp plans to support an optional memory tier that reuses VRAM from secondary GPUs that are not suitable for inference. A GPU may be too old for the selected compute backend, lack required tensor instructions, or otherwise be excluded from model execution while still having a working driver path capable of allocating and transferring data to its VRAM.

That VRAM can potentially be useful as temporary spill storage when the active inference GPU is under memory pressure.

This feature is **not literally zram**. Linux zram stores compressed pages in system RAM. The feature described here is a **VRAM-backed memory/swap tier**. It may be used alongside ordinary RAM, zram, and disk swap, but those are separate mechanisms.

## Intended memory hierarchy

A typical configuration may look like:

```text
Active inference GPU VRAM
        |
        v
Legacy / spare GPU VRAM
        |
        v
System RAM / compressed RAM (zram)
        |
        v
Disk-backed swap or model reload
```

The exact order is policy-driven. On some machines host RAM or zram may be faster than a spare GPU connected through a narrow PCIe link, so the memory oversubscription handler must be allowed to choose a different ordering.

## What qualifies as a legacy/spare GPU

A device does not have to be capable of running EasyDifussion-Cpp inference to participate. It only needs a safe, working mechanism for allocating VRAM and copying bytes between that VRAM and a usable host or device address space.

Examples include:

- a GPU whose architecture is unsupported by the selected inference kernels;
- a secondary GPU intentionally excluded from inference;
- a compute accelerator whose memory API remains usable even though the model backend does not support it;
- a display GPU with genuinely spare VRAM, provided a conservative display reserve is maintained.

A GPU with no functioning operating-system/driver access does **not** qualify. The project must not attempt unsafe raw physical-memory access merely to recover otherwise inaccessible VRAM.

## Relationship to inference backends

Memory-tier eligibility and inference eligibility are independent properties.

A device being accepted as a VRAM tier must never cause it to be advertised as a CUDA, ROCm, oneAPI, Vulkan, OpenGL, OpenCL, DirectML/D3D12, or other compute device. Conversely, an inference-capable device should not automatically be consumed as swap space.

The memory manager must maintain separate device roles:

```text
INFERENCE       executes model operations
MEMORY_TIER     stores evicted/staged allocations
DISPLAY         reserved for graphics/display duties
DISABLED        not used by EasyDifussion-Cpp
```

A device may only have multiple roles when an explicit coexistence policy reserves capacity for each role.

## Backend access methods

The implementation should use supported allocation APIs whenever possible. Candidate adapters include:

- CUDA Driver API allocations for NVIDIA devices;
- HIP/ROCm allocations for supported AMD devices;
- SYCL/oneAPI USM or device allocations;
- OpenCL buffers;
- Vulkan device memory;
- D3D12 heaps/resources on Windows;
- other documented driver interfaces that provide bounded allocation and transfer operations.

No individual API is required for the architecture. The common VRAM-tier interface sits above these adapters.

## Core interface

Each adapter should provide the equivalent of:

```text
probe_device()
capacity()
free_capacity()
allocate(bytes)
free(handle)
copy_to_tier(handle, source, bytes)
copy_from_tier(destination, handle, bytes)
synchronize()
health()
shutdown()
```

The oversubscription handler should not need to know whether a tier is implemented through CUDA, OpenCL, Vulkan, D3D12, or another API.

## Integration with the memory oversubscription handler

The memory oversubscription handler (TODO 33) owns the global decision about when and where data moves. This feature only provides an additional destination/source tier.

When active-GPU VRAM approaches a configured high-water mark, the handler may identify allocations that can be evicted and move them to a spare GPU. Before those allocations are needed again, they are copied back to an appropriate compute-visible location.

The handler must avoid repeated ping-pong migration. High- and low-water marks, minimum residency times, transfer cost, allocation priority, and predicted reuse should be considered.

## Performance considerations

Spare VRAM is not automatically faster than system RAM.

Important factors include:

- PCIe generation and lane width;
- whether both GPUs share a PCIe root complex;
- peer-to-peer transfer availability;
- NUMA placement;
- host-memory bandwidth;
- zram compression ratio and CPU cost;
- allocation size;
- frequency with which an evicted allocation is reused.

Before `auto` mode uses a GPU, EasyDifussion-Cpp should benchmark or estimate its transfer characteristics. A GPU on PCIe x1 may be useful for cold model weights but harmful for frequently reused tensors.

## Display GPU protection

Display GPUs require special handling. Reported free VRAM cannot be treated as fully available because the desktop compositor, applications, browsers, games, and the driver may allocate additional memory at any time.

The implementation must support a configurable reserve and should use a conservative automatic reserve by default. If display memory pressure rises, memory-tier allocations should be evicted before interfering with the desktop.

## Multiple spare GPUs

Multiple devices may form multiple tiers. Each should be tracked independently with:

- stable device identity;
- usable capacity;
- reserved capacity;
- occupied capacity;
- measured transfer bandwidth;
- PCIe/NUMA topology where available;
- current health;
- failure count;
- bytes transferred in and out.

The manager may rank devices by cost rather than simply filling GPU 0 and then GPU 1.

## Failure handling

VRAM tier contents are temporary. They must never be treated as persistent storage.

If a tier device resets, disappears, becomes unhealthy, or fails a transfer, the handler must stop placing new allocations on it. Recoverable data should be reconstructed from another copy, reloaded from the model, or restored from a lower tier. If the only valid copy is lost, the current operation should fail cleanly rather than continue with corrupted tensor data.

A failed tier must not crash unrelated inference devices whenever isolation is possible.

## Linux block/swap experiments

A Linux implementation may optionally investigate exposing reserved VRAM through a block-device/NBD-style mechanism and allowing the kernel to use it as swap. This is an experimental alternative to application-managed tensor paging, not the default architecture.

Kernel-managed swap and application-managed model paging solve different problems. EasyDifussion-Cpp should prefer application-managed movement when it knows tensor reuse and lifetime information that the kernel cannot see.

If a block-device mode is implemented, it must be explicitly enabled, cleanly detached at shutdown, and isolated from VRAM used for display or inference.

## Configuration concept

A future configuration may resemble:

```yaml
memory:
  oversubscription: auto
  legacy_vram_tier:
    mode: off        # off | auto | manual
    devices: []      # stable device identifiers for manual mode
    display_reserve_mb: 2048
    benchmark_on_start: true
    minimum_bandwidth_gbps: 0
```

The exact configuration schema is not final. Until the implementation is proven safe, the feature should default to `off`.

## Telemetry

When enabled, diagnostics should clearly report:

```text
Legacy VRAM tier: enabled
Device: <stable identifier / model>
Role: MEMORY_TIER
Capacity reserved: ...
Currently occupied: ...
Host -> tier bandwidth: ...
Tier -> host bandwidth: ...
Bytes migrated: ...
Evictions: ...
Failures: ...
Fallback tier: ...
```

This information is important because a successful allocation does not necessarily mean the tier improves performance.

## Security and correctness boundaries

The implementation must not:

- access undocumented physical VRAM addresses by default;
- overwrite framebuffer or driver-owned memory;
- assume VRAM survives reset, suspend, reboot, or process termination;
- silently enable system swap;
- use an unsupported device without a functioning driver/API allocation path;
- allow two allocators to unknowingly claim the same VRAM capacity;
- report memory-tier capacity as inference VRAM.

## Development stages

### Stage 1 — Discovery
Enumerate unused GPUs and report whether a safe memory-allocation adapter exists. Do not allocate persistent tier memory.

### Stage 2 — Allocation test
Reserve a bounded amount of VRAM, perform deterministic copy-in/copy-out verification, release it, and confirm no residual allocation remains.

### Stage 3 — Memory-tier API
Expose the backend-neutral allocation/copy interface and telemetry.

### Stage 4 — Oversubscription integration
Allow TODO 33's handler to migrate selected cold allocations into and out of the tier.

### Stage 5 — Multi-GPU policy
Rank multiple spare devices by capacity, topology, bandwidth, and health.

### Stage 6 — Long-duration validation
Test repeated model loads, generation workloads, memory pressure, display use, device failure, and shutdown/restart cycles.

## Completion criteria

The feature is ready for normal use when an inference-unsupported but driver-accessible GPU can safely hold evicted model data, restore that data without corruption, coexist with the active inference/display devices, recover or fail cleanly when the tier disappears, demonstrate useful behavior under real memory pressure, and leave no residual allocations or system resources after shutdown.

See also:

- `TODO.md` — project checklist;
- `docs/todo/33-memory-oversubscription-handler.md` — global memory pressure and migration policy;
- `docs/todo/34-legacy-gpu-vram-swap-tier.md` — implementation checklist and acceptance criteria.
