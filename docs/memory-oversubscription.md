# RAM and VRAM Oversubscription

## Overview

EasyDifussion-Cpp plans to support a backend-aware memory oversubscription system that allows workloads to continue when their preferred memory tier is too small, while avoiding uncontrolled out-of-memory crashes or system-wide swap thrashing.

Oversubscription means allowing the logical working set of a workload to exceed the immediately available capacity of one memory tier. For GPU inference this most commonly means a model or its working tensors exceed dedicated VRAM and some data must reside elsewhere. The same principle also applies to system RAM: a workload can exceed a configured RAM budget and require compressed memory, disk-backed storage, model reload, or another tier.

Oversubscription does **not** create free memory. It trades capacity for transfer, migration, compression, recomputation, and synchronization cost.

## Memory tiers

The architecture should treat memory as explicit tiers rather than as one combined number:

```text
Tier 0: Active GPU / accelerator local VRAM
Tier 1: Other usable GPU VRAM (optional legacy/spare VRAM tier)
Tier 2: Pinned or ordinary system RAM
Tier 3: Compressed RAM / zram where available
Tier 4: Disk-backed swap, mapped model files, or reloadable storage
```

The ordering is not universally fixed. The memory manager should rank available destinations according to backend capabilities and measured cost. For example, fast host RAM may outperform spare VRAM behind PCIe x1, while peer-connected GPU VRAM may outperform host staging.

## VRAM oversubscription

VRAM oversubscription occurs when the logical GPU working set is larger than the dedicated memory that can safely remain resident.

The handler should avoid waiting for a failed allocation. It should monitor a usable VRAM budget and transition through pressure states before physical exhaustion:

```text
NORMAL -> PRESSURE -> CRITICAL -> RECOVERY
```

A possible default policy is:

- NORMAL: below the configured high-water mark; no forced migration.
- PRESSURE: reclaim disposable caches and avoid unnecessary new residency.
- CRITICAL: migrate/offload eligible cold allocations before another large allocation.
- RECOVERY: restore normal placement only after usage falls below a lower threshold.

Separate entry and exit thresholds provide hysteresis and prevent tensors from repeatedly bouncing between VRAM and RAM.

## RAM oversubscription

Host RAM requires its own budget. Moving everything out of VRAM is not a valid solution if doing so exhausts system memory.

The handler should track:

- physical RAM;
- RAM currently available to the process;
- configured EasyDifussion-Cpp host-memory limit;
- pinned/page-locked memory, which should be budgeted separately because excessive pinning harms the OS;
- zram/compressed-memory availability;
- operating-system swap availability and current pressure;
- memory already committed by other EasyDifussion-Cpp processes/components.

The project should never intentionally consume all system RAM. A configurable operating-system reserve must remain available.

When the host-memory budget is approached, the handler may compress, evict, remap, reload, or move sufficiently cold data to a lower tier instead of allowing the kernel OOM killer or Windows memory pressure to make the decision unexpectedly.

## Allocation classes

Not all memory has equal value or migration cost. Every significant allocation should be classified where practical.

Suggested classes include:

```text
MODEL_WEIGHT       persistent, expensive to recreate but often cold by layer
CACHE              persistent but potentially discardable/rebuildable
ACTIVATION         temporary and generally latency-sensitive
WORKSPACE          temporary scratch allocation
INTERMEDIATE       encode/decode/control intermediate
OUTPUT             user-visible result data
MAPPED_MODEL       reloadable from a mapped model file
RECOMPUTABLE       can be discarded and regenerated
```

The classification gives the policy engine enough information to choose sensible eviction candidates.

## Eviction priority

A general reclaim sequence should prefer the cheapest actions first:

1. release already-unused allocations;
2. drop disposable caches;
3. release/recreate temporary workspaces;
4. evict explicitly recomputable data;
5. migrate cold persistent allocations to a faster lower tier;
6. migrate additional model state if required;
7. use compressed RAM or mapped/disk-backed storage when configured;
8. return a controlled OOM if no safe tier remains.

The exact sequence can vary by workload and backend.

## Backend-specific behavior

### DirectML / D3D12

Investigate D3D12 residency, budget reporting, shared GPU memory, pageable resources, `Evict`/`MakeResident`-style residency control, and DirectML resource requirements. Windows memory budgeting should be treated as authoritative rather than assuming reported dedicated VRAM is the only usable capacity.

### CUDA

Distinguish several different mechanisms rather than treating all CUDA memory as equivalent:

- ordinary device allocations plus explicit host offload;
- pinned host staging;
- managed/unified memory where appropriate;
- Linux HMM/ATS capabilities when available;
- peer GPU memory when topology and driver support make it useful.

A CUDA backend must probe capabilities instead of assuming unified-memory oversubscription behaves identically across operating systems, drivers, and GPU generations.

### ROCm / HIP

Probe HMM/XNACK/unified-memory capabilities and otherwise use explicit HIP host/device staging. Unsupported combinations must fall back conservatively.

### oneAPI / SYCL

Use SYCL/Level Zero capability discovery to distinguish device, host, and shared USM. Shared USM availability alone does not guarantee that every allocation should use it; performance and migration behavior still need validation.

### Vulkan

Use explicit memory-budget information and staging strategies. Do not assume device-local allocations can transparently spill into host memory. Heap properties and driver behavior must be inspected.

### OpenCL

Use device/host buffers and supported shared-memory capabilities where available, with explicit copies as the portable fallback.

### OpenGL

Treat OpenGL oversubscription conservatively. Explicit buffer residency and staging policy should be preferred over relying on undocumented driver eviction behavior.

## Spare / legacy GPU VRAM

TODO 34 introduces an optional additional tier using VRAM from GPUs that are not selected for inference. This is separate from ordinary VRAM oversubscription but can be used by the same policy engine.

For example:

```text
RTX inference VRAM -> old GPU VRAM -> host RAM -> zram -> disk
```

or, if host RAM is faster:

```text
RTX inference VRAM -> host RAM -> old GPU VRAM -> disk
```

The manager should choose according to measured cost rather than assuming GPU memory is always the better tier.

## Multi-GPU oversubscription

Each GPU has an independent memory budget. One full GPU must not cause unrelated allocations to be evicted from another GPU unless the scheduler deliberately chooses cross-device placement.

Track per device:

- total physical memory;
- driver-reported budget;
- committed memory;
- EasyDifussion-Cpp-owned memory;
- reserved safety margin;
- reclaimable memory;
- migration destinations;
- peer-access capabilities;
- topology/bandwidth information.

A multi-GPU K-sampling or pipeline configuration may have very different memory pressure on each device, so global percentage-only accounting is insufficient.

## Memory ownership

Every allocation must have one authoritative state. Migration must never leave two writable copies that the runtime assumes are both current.

A conceptual state machine may be:

```text
RESIDENT_DEVICE
    -> MIGRATING_OUT
    -> RESIDENT_TIER
    -> MIGRATING_IN
    -> RESIDENT_DEVICE
```

Read-only replicated weights may permit multiple valid copies, but replication must be represented explicitly.

Cancellation, model unload, backend reset, and process shutdown must be able to interrupt or drain migrations without leaking either copy.

## Allocation retry

An allocation failure should not cause an unbounded retry loop.

A bounded sequence should be used:

```text
allocation request
    |
    +-> success
    |
    +-> insufficient budget
          |
          +-> reclaim / migrate
          |
          +-> retry once (or a small configured bound)
                |
                +-> success
                +-> controlled OOM with diagnostics
```

The error should explain the requested allocation size, backend/device, available budget, reclaim attempt, and lower tiers considered.

## Policy modes

A future user-facing setting may provide:

```text
Disabled      Never intentionally oversubscribe.
Conservative  Reclaim caches and offload only clearly cold/reloadable data.
Balanced      Permit planned migration of model state and use safe native mechanisms.
Aggressive    Maximize capacity using configured lower tiers, accepting greater latency.
```

Additional limits should include reserve VRAM, maximum host RAM, maximum pinned RAM, allowed spare-GPU tiers, zram/disk permission, and maximum migration latency or bandwidth cost.

## Example configuration

The exact schema is not final, but a future configuration may resemble:

```yaml
memory:
  oversubscription:
    mode: balanced
    vram_high_watermark: 0.88
    vram_low_watermark: 0.72
    reserve_vram_mb: 1024
    max_host_ram_mb: 32768
    host_ram_reserve_mb: 4096
    allow_zram: true
    allow_disk: false
    allocation_retries: 1

  legacy_vram_tier:
    mode: off
```

Defaults should prioritize safety over maximum theoretical capacity.

## Telemetry

Useful diagnostics include:

```text
Device budget / committed / available
Host RAM budget / committed / available
Pressure state
Allocation class and size
Bytes reclaimed
Bytes migrated VRAM -> RAM
Bytes migrated RAM -> VRAM
Bytes migrated through spare GPU VRAM
Compression / disk activity
Migration latency and bandwidth
Allocation retries
Controlled OOM count
Backend mechanism used
```

Telemetry should distinguish driver-reported usage from memory directly owned by EasyDifussion-Cpp.

## Performance

Oversubscription is expected to become slower as the working set exceeds fast local memory. The goal is graceful degradation rather than pretending capacity is free.

Performance depends on:

- tensor reuse frequency;
- PCIe bandwidth and topology;
- host-memory bandwidth;
- compression ratio and CPU cost;
- page migration behavior;
- GPU/CPU synchronization;
- whether the model can stream layer-by-layer;
- whether data can be cheaply recomputed instead of transferred.

The policy should eventually use measured behavior to avoid migrating hot tensors.

## Failure modes to prevent

The implementation must specifically guard against:

- VRAM allocation retry loops;
- RAM exhaustion caused by excessive GPU offload;
- Linux OOM-killer activation caused by an unlimited host tier;
- Windows commit exhaustion;
- zram or disk swap thrashing;
- excessive pinned-memory allocation;
- migration ping-pong around a threshold;
- stale tensor copies;
- freeing the source before a transfer is complete;
- multi-GPU accounting mistakes;
- treating driver-managed/shared memory as dedicated free VRAM;
- silent performance collapse without telemetry.

## Testing strategy

Testing should deliberately create pressure rather than waiting for accidental OOM conditions.

Recommended cases:

1. workload fits comfortably in VRAM;
2. workload approaches the high-water mark;
3. workload exceeds dedicated VRAM slightly;
4. workload substantially exceeds VRAM but fits in host RAM;
5. workload exceeds configured host RAM and must use another allowed tier or fail;
6. repeated pressure/recovery cycles;
7. model cancellation during migration;
8. model unload while allocations are offloaded;
9. one pressured GPU in a multi-GPU system;
10. spare GPU tier enabled and disabled;
11. lower tier disappears or fails;
12. oversubscription disabled, confirming legacy allocation behavior is unchanged.

Correctness checks should accompany performance measurements. A slower correct result is preferable to a fast corrupted result.

## Relationship to existing TODOs

- `docs/todo/33-memory-oversubscription-handler.md` defines the implementation checklist and completion criteria for the central handler.
- `docs/todo/34-legacy-gpu-vram-swap-tier.md` defines the optional spare-GPU VRAM tier.
- `docs/legacy-gpu-vram-memory-tier.md` explains the spare/legacy GPU feature in detail.
- `docs/todo/15-directml-d3d12-backend.md` covers the Windows DirectML/D3D12 backend whose residency model is especially relevant to Windows oversubscription.

## Completion goal

The final system should allow EasyDifussion-Cpp to know when VRAM or RAM is approaching exhaustion, choose a safe lower memory tier or reclaim strategy, migrate only eligible data, recover that data correctly, expose the performance cost, and return a controlled diagnostic OOM when no safe capacity remains.
