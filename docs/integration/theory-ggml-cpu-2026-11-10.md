# Theory ownership adapter on real GGML CPU buffers

## Runtime integration

SDKIT_BUILD_THEORY_CPU_RESOURCES is an opt-in, default-OFF build option.
It builds a static resource adapter using the existing ggml and ggml-cpu targets
already configured by SDKit. No second GGML source tree, runtime, device scan,
GPU selection or inference model is introduced.

The native CPU allocator utility uses ggml_backend_cpu_buffer_type,
ggml_backend_buft_alloc_buffer, get_base/get_size and buffer_free. It shares
the existing move-only facade, Common routing and allocation-budget contracts.
The direct host test build still uses malloc/free; storage_backend() identifies
which implementation is actually executing.

This is real CPU backend-buffer execution, not model parsing, tensor-graph
execution or generation. Neither UI nor existing model ownership is replaced.

## Allocation identity and truthful CPU release

A synchronized native allocation table records each owned pointer and a unique,
nonzero, monotonically increasing identity. Unload validates both fields before
releasing through the allocator that created the buffer. Stale, forged, foreign
or already released descriptors fail without freeing the live allocation.
Identities are never reused; identity exhaustion rejects allocation.

The native allocator utility owns resources with RAII before registration.
Allocation-table insertion failures cannot orphan a buffer. The concurrency
fixture also requires zero tracked live allocations after all owners release.

## Translation exceptions

Load callbacks now receive a provisional result by reference. They must publish
ownership immediately after allocation, before subsequent throwing operations.
Common catches standard and unknown exceptions and rolls back any published
ownership through the originating translation. The CPU native allocator handles
pre-publication failures with RAII.

Nine synthetic failure cases cover malformed metadata, failure status and
exceptions before/after allocation. Allocation and rollback counts match.
This contract cannot repair a callback that hides an allocation instead of
publishing ownership; future providers still require review.

## Module lifetime

Both registration tables accept an optional shared module-lifetime guard.
Immutable entries retain it, and dispatch copies the guard while invoking a
callback. The fixture drops the caller's reference and verifies the registry
continues to pin the module owner. Process-linked CPU callbacks need no guard.

There is no unregister or hot-unload API. Dynamically supplied callbacks must
provide a guard that actually owns their code/module lifetime. The test exercises
the lease mechanism, not an OS dlopen/dlclose loader. Real plugin loading and
controlled shutdown are separate work; do not advertise hot reload.

## Verification

- Five host contract tests pass normally, under ThreadSanitizer and under
  ASan/UBSan with leak detection disabled for the ptrace executor.
- Four CTest executables pass against the actual GGML CPU adapter: public facade,
  lifecycle/identity, ownership/concurrency and malformed/throwing load.
- The public executable reports Allocator: ggml-cpu.
- Lifecycle tests reject wrong allocation IDs and stale descriptors.
- Ownership stress runs eight threads and 1,024 cycles, ending with zero
  tracked native allocations.
- The GGML libraries themselves were not rebuilt with sanitizers for this run;
  do not transfer host-test sanitizer coverage to that dependency build.
- No GPU/model inference or packaged application test was run.

Build using an explicitly CPU-configured SDKit CMake build, then enable
SDKIT_BUILD_THEORY_CPU_RESOURCES and build target theory-cpu-resource-tests.
Run CTest with the filter ^theory-.

## Remaining boundary

This proves bounded CPU backend-buffer ownership, not integration of model
weights, GGML tensor graphs, scheduler ownership, offload or heterogeneous devices.
An actual model adapter must preserve those runtimes' existing allocation and
lifetime contracts. Unknown providers' unload status is still a contract; the
native CPU implementation is the one validated by the identity table here.
