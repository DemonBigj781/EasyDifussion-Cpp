# oneAPI / SYCL native-call inventory for xFormers

Status: raw inventory. Do not choose normalized translation names from this file yet.

Purpose: record SYCL-native execution, memory, synchronization, datatype, math, queue, capability, and error concepts that may be required by an xFormers-compatible implementation. SYCL is not treated as CUDA with renamed functions.

## Execution and compilation model

- Device work is expressed through C++ kernel submissions rather than triple-chevron syntax.
- `sycl::queue` is the principal host-side submission/execution object.
- `queue.submit(...)` creates a command group through a `sycl::handler`.
- `handler::parallel_for(...)` submits kernels.
- `sycl::range` supports simple N-dimensional iteration.
- `sycl::nd_range` combines explicit global and local ranges when work-group coordination is required.
- `sycl::nd_item` exposes work-item, work-group, and sub-group execution identity.
- Kernel lambdas/functors capture parameters into device code subject to SYCL device-copy rules.

## Queue and dependency model

Relevant native concepts include:

- in-order versus out-of-order queues;
- event-returning submissions;
- explicit event dependencies;
- handler dependency declarations;
- queue/device/context ownership;
- asynchronous exception handling.

Potential primitives to record when used:

- `sycl::event`
- `event::wait()` / `wait_and_throw()`
- `queue::wait()` / `wait_and_throw()`
- handler dependency APIs

A future xFormers translation must document whether ordering is inherited from queue configuration or established by explicit events.

## Work-item, group, and sub-group identity

Relevant `sycl::nd_item` information includes:

- `get_global_id(...)`
- `get_global_linear_id()`
- `get_local_id(...)`
- `get_local_linear_id()`
- `get_group(...)` / group ID access
- local/global range access
- `get_group()` for work-group operations
- `get_sub_group()` for sub-group operations

Sub-group width is device/compiler dependent unless explicitly constrained.

## Work-group synchronization

Relevant synchronization concepts include:

- `sycl::group_barrier(item.get_group())` for work-group synchronization;
- sub-group barriers for sub-group-local synchronization where supported;
- explicit memory scope/order on atomic operations;
- event/queue synchronization on the host side.

A simple `range` kernel does not expose the same coordinated local-work-group model as an `nd_range` kernel, which is important for attention reductions.

## Local/shared storage

- `sycl::local_accessor<T, Dimensions>` provides explicit work-group-local storage.
- Local accessor storage is allocated per work-group for a command group/kernel submission.
- Work-items coordinate access using barriers.
- Local-memory size is device-dependent and should be queried/validated for the selected tile/work-group size.

Alternative USM/local mechanisms should be documented separately if used by the implementation.

## Address spaces and memory models

SYCL may use buffer/accessor or USM programming styles.

Relevant concepts include:

- device/global memory;
- local memory via `local_accessor`;
- private per-work-item storage;
- buffers/accessors where used;
- USM device allocations;
- USM shared allocations;
- USM host allocations;
- explicit queue copies/memcpy/fill operations;
- pointer/address-space rules in device code.

The first real oneAPI xFormers implementation must record which model it chooses; no common translation requirement should be inferred before that choice.

## Sub-group collectives and cross-lane behavior

Candidate SYCL-native primitives for reductions/exchange include:

- `sycl::reduce_over_group(...)`
- `sycl::exclusive_scan_over_group(...)`
- `sycl::inclusive_scan_over_group(...)`
- `sycl::group_broadcast(...)`
- `sycl::select_from_group(...)`
- `sycl::shift_group_left(...)`
- `sycl::shift_group_right(...)`
- `sycl::permute_group_by_xor(...)`

Availability/performance can depend on target/device/compiler. None are yet claimed as required by this repository's xFormers path.

## Atomics and memory ordering

- `sycl::atomic_ref` provides atomic operations with explicit:
  - memory order;
  - memory scope;
  - address space.

Relevant operation categories include fetch-add, exchange, compare-exchange, min/max where supported, and bitwise operations for integral types.

This explicit memory-model structure is a key native property to preserve during later comparison instead of flattening all atomics into one CUDA-style spelling.

## Scalar and vector datatypes

Relevant type categories include:

- F32 through standard `float`;
- FP16 through `sycl::half`;
- BF16 through oneAPI/SYCL extension types where supported by the selected toolchain/device;
- integer scalar types;
- vector types or packed storage where an optimized implementation uses them.

Exact BF16 type names, conversion APIs, and device support must be recorded from the actual implementation and compiler version.

## Math primitives relevant to attention

SYCL device math provides or permits equivalents for categories including:

- exponential;
- maximum/minimum;
- tangent/hyperbolic tangent;
- reciprocal/division;
- finite/NaN/infinity classification;
- `sqrt` for default scale calculation;
- `floor`, `log2`, and `pow` for host/device parameter preparation if needed.

Exact namespace/function choices must be captured from the real source.

## Dot-product and reduction strategy space

Possible oneAPI/SYCL strategies include:

- work-item scalar accumulation plus `local_accessor` reduction;
- sub-group reduction;
- hierarchical sub-group + work-group reduction;
- joint/reduction algorithms;
- vendor matrix extensions;
- oneMKL/library-backed matrix operations;
- fused streaming attention.

This file does not select a strategy. The real implementation must record work-group dimensions, subgroup assumptions, accumulator precision, and local-memory requirements.

## Matrix hardware / vendor extensions

Intel GPU targets may expose matrix acceleration through oneAPI/SYCL extensions. Relevant categories to inventory only if used include:

- joint-matrix APIs/extensions;
- DPAS-oriented compiler extensions;
- oneMKL calls;
- architecture-specific tile/sub-group requirements.

These may be more appropriate than mechanically reproducing a CUDA shared-memory reduction.

## Mask, bias, and ALiBi behavior to inventory

The real implementation must specify:

- additive mask representation;
- causal mask behavior if present;
- mask datatype/layout and broadcasting;
- ALiBi parameter preparation and application;
- logit soft-cap;
- attention sinks;
- fully masked rows;
- NaN and infinity behavior.

## Softmax strategy to inventory

Possible forms include:

- materialized stable softmax;
- online softmax;
- sub-group reduction;
- work-group reduction via local memory;
- fused normalization and AV.

Record accumulator dtype, reduction topology, special-value behavior, and whether score storage is materialized when implementation exists.

## Kernel/device capability queries

Relevant SYCL device information may include:

- device type/vendor/name;
- maximum work-group size;
- local-memory capacity;
- supported subgroup sizes;
- FP16 capability;
- BF16/extension capability;
- USM capabilities;
- global memory size;
- preferred/native vector widths;
- architecture/backend identity exposed by the oneAPI runtime.

Only queried properties actually used for dispatch should later become translation inputs.

## Error and exception behavior

SYCL differs strongly from CUDA/HIP here.

Relevant native mechanisms include:

- synchronous C++ exceptions during submission/API calls;
- asynchronous exceptions delivered through an async handler or surfaced by `wait_and_throw()`;
- event status/dependency failures;
- backend-specific errors wrapped by SYCL implementations.

The eventual translation must document exactly where exceptions are caught and how they become Common diagnostics/status without exposing SYCL exception types to Library callers.

## Architecture-specific facts to preserve

- Sub-group size is not universally fixed.
- Work-group size and local-memory capacity are device-dependent.
- Queue/event ordering is not represented by CUDA/HIP stream types.
- Submission is command-group based, not direct launch syntax.
- Device capability can vary substantially between Intel CPU, Intel GPU, and other SYCL targets.
- An equivalent oneAPI implementation may use a fundamentally different kernel decomposition while producing the same attention semantics.

## Current repository state

The oneAPI xFormers `definition/` tree contains structural method placeholders for CPU/GPU targets, while the `translation/` CPU/GPU/NPU directories are currently empty placeholders.

Therefore this document is a native capability inventory, not a runtime implementation description.

## Inventory gaps to fill from the first real oneAPI implementation

- queue creation/ownership source;
- in-order/out-of-order behavior;
- dependency/event strategy;
- kernel submission form;
- global/local range geometry;
- subgroup assumptions;
- local-memory use;
- Q/K/V layout requirements;
- dtype and conversion types;
- mask/bias/ALiBi handling;
- reduction/softmax strategy;
- output datatype;
- exception/error normalization path;
- architecture capability gates;
- runtime validation results by device family.

## Questions reserved for the later comparison phase

Do not answer these here.

- Which SYCL group operations are true semantic equivalents of CUDA/HIP block or warp primitives?
- Which queue/event concepts need a translation-level execution object rather than being hidden inside backend code?
- Should subgroup width be visible as capability metadata to the translation layer?
- Which local-memory/reduction behaviors are correctness requirements versus implementation choices?
- Which matrix extensions should remain entirely backend-private?
