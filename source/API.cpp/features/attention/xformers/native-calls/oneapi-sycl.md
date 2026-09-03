# oneAPI / SYCL native-call inventory for xFormers

Status: raw inventory. Do not choose normalized translation names from this file yet.

This file records SYCL-native primitives that can express the categories of execution needed by memory-efficient attention. SYCL uses a different execution model and naming scheme than CUDA/HIP, so semantic comparison must happen later.

## Execution and launch

- `sycl::queue` represents the submission/execution queue.
- `queue.submit(...)` submits a command group.
- `handler::parallel_for(...)` launches work.
- `sycl::nd_range` provides explicit global and local ranges when work-group coordination is required.
- `sycl::nd_item` provides per-work-item execution identity and work-group access.

## Work-item / group identity

Relevant `nd_item` information includes:

- global ID / linear global ID;
- local ID / linear local ID;
- group ID;
- local range and global range;
- group/sub-group accessors where applicable.

Exact accessor spelling should follow the SYCL version/compiler selected by the oneAPI implementation.

## Synchronization

- `sycl::group_barrier(item.get_group())` provides a work-group barrier.
- Work-group barriers require an `nd_range`/`nd_item` style kernel; simple range-based `parallel_for` forms do not provide the same work-group-local synchronization model.
- Sub-group operations/barriers are available where the selected device exposes appropriate sub-group support.

## Local/shared storage

- `sycl::local_accessor` is the principal work-group-local storage mechanism for explicit local memory in SYCL command groups.
- Its lifetime and access model differ syntactically from CUDA/HIP `__shared__`, so later translation must compare behavior rather than names.

## Atomics and ordering

- `sycl::atomic_ref` provides atomic operations with explicit memory order, memory scope, and address-space parameters.
- SYCL exposes explicit memory-order and memory-scope concepts that may map to several different CUDA/HIP fence/atomic variants rather than a single spelling.

## Sub-group / cross-lane operations

SYCL provides sub-group collectives and value movement operations. Candidate primitives for attention reductions or exchange include:

- sub-group reductions;
- broadcast/select-style value exchange;
- shuffle/permutation facilities where supported by the implementation/extensions in use.

The exact subset needed must be recorded from the actual oneAPI implementation rather than inferred from CUDA warp code.

## Scalar and dtype considerations

- standard C++ scalar types;
- `sycl::half` for FP16;
- bfloat16 support depends on the oneAPI/SYCL implementation and extension/type support chosen by the project;
- math functions are supplied through SYCL/C++ device math facilities rather than CUDA-specific intrinsics.

## Architecture-specific facts to preserve

- Sub-group size is not universally fixed; implementation/device capabilities must be queried or constrained explicitly.
- Work-group size and local-memory availability are device-dependent.
- Queue/event ordering semantics are not expressed through CUDA/HIP stream types.
- SYCL kernel submission is host-side command-group construction rather than CUDA/HIP's direct triple-chevron source syntax.
- An equivalent xFormers implementation may use a very different kernel structure while preserving the same mathematical behavior.

## Not yet claimed

This inventory does not claim specific SYCL calls are exact equivalents of CUDA or HIP calls. It exists so those comparisons can be made explicitly later.
