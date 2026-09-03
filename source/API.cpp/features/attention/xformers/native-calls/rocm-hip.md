# ROCm / HIP native-call inventory for xFormers

Status: raw inventory. Do not choose normalized translation names from this file yet.

Purpose: record HIP/ROCm-native execution, memory, synchronization, datatype, math, launch, validation, and error primitives that may be required by an xFormers-compatible implementation. Similar CUDA-like spelling does not imply identical execution semantics.

## Execution and compilation model

- `__global__` marks HIP device kernel entry points.
- `__device__` and `__host__` are available with CUDA-like source semantics.
- Device helpers may use `__forceinline__` or compiler-specific inlining attributes.
- HIP source can often preserve CUDA-style template dispatch while targeting AMD architectures.
- Compile-time target macros and architecture checks may select AMD-specific behavior.

## Kernel launch and execution objects

- CUDA-style triple-chevron kernel launch is supported by HIP.
- `hipLaunchKernelGGL(...)` provides an explicit HIP launch form.
- `dim3` provides grid/block geometry.
- `hipStream_t` represents an ordered asynchronous HIP execution stream.
- HIP events and stream-wait operations exist for explicit dependency management when needed.
- Kernel execution remains asynchronous relative to the host unless synchronization is requested.

## Thread, block, grid, warp/wave identity

HIP device code exposes CUDA-like built-ins:

- `threadIdx.x/y/z`
- `blockIdx.x/y/z`
- `blockDim.x/y/z`
- `gridDim.x/y/z`
- `warpSize`

Important AMD-specific fact: the hardware execution group is a wavefront and its width is architecture-dependent. Code must not infer NVIDIA warp width from similar source syntax.

## Shared/LDS storage

- `__shared__` maps to block/work-group-local storage, backed by AMD LDS on GPU targets.
- Lifetime is scoped to the executing block/work-group.
- Static and dynamic shared-memory usage may be selected depending on kernel structure.
- Capacity, bank behavior, and occupancy impact are architecture-specific and should be recorded when an actual ROCm xFormers kernel is implemented.

## Work-group synchronization

- `__syncthreads()` provides block/work-group synchronization.
- HIP also exposes:
  - `__syncthreads_count(...)`
  - `__syncthreads_and(...)`
  - `__syncthreads_or(...)`

These are raw native capabilities; the future translation vocabulary must not assume that a CUDA block and AMD work-group are performance-equivalent simply because the barrier spelling matches.

## Wavefront/cross-lane operations

HIP provides shuffle/cross-lane primitives such as:

- `__shfl(...)`
- `__shfl_up(...)`
- `__shfl_down(...)`
- `__shfl_xor(...)`
- sync-style variants where available in the selected ROCm version

Vote/ballot-style operations are also available in HIP. AMD mask width and wavefront behavior must be documented from the actual target architecture rather than copied from CUDA assumptions.

## Memory fences and visibility

HIP provides:

- `__threadfence_block()`
- `__threadfence()`
- `__threadfence_system()`

These are distinct from a work-group barrier and should remain distinct inventory entries.

## Memory/address-space behavior

Relevant native concepts include:

- global device memory through ordinary device pointers;
- block/work-group-local LDS through `__shared__`;
- register/private values local to a work-item;
- byte-stride or typed pointer arithmetic for tensor layouts;
- host/device copies and managed/pinned memory through HIP runtime APIs when integration requires them.

The actual ROCm xFormers implementation should document whether it consumes ggml byte strides directly, requires contiguous rows, or performs staging/repacking.

## Stream/event ordering primitives

Potential primitives to inventory when used by the real implementation:

- `hipStreamCreate*`
- `hipStreamSynchronize`
- `hipDeviceSynchronize`
- `hipEventCreate*`
- `hipEventRecord`
- `hipEventSynchronize`
- `hipStreamWaitEvent`

These are not yet claimed as xFormers requirements.

## Atomics

HIP provides native atomic families including:

- `atomicAdd`
- `atomicSub`
- `atomicMin`
- `atomicMax`
- `atomicExch`
- `atomicCAS`
- bitwise atomic operations

Floating-point atomic availability/performance may vary by architecture and datatype. The current CUDA xFormers implementation does not require atomics, so no future translation requirement should be inferred from their presence here.

## Scalar and packed datatypes

Relevant ROCm/HIP datatype categories include:

- F32 scalar values;
- HIP half types and half conversion intrinsics;
- ROCm/HIP bfloat16 types where supported by the chosen headers/toolchain;
- packed/vector half/bfloat forms where an optimized implementation uses them;
- integer bit representations used by explicit software conversion paths.

The exact type names and conversion intrinsics must be taken from the implementation actually chosen for this repository.

## Math functions relevant to attention

Device math categories likely required by an xFormers-compatible path include:

- exponential;
- maximum/minimum;
- hyperbolic tangent for logit soft-cap;
- finite/NaN/infinity handling;
- reciprocal/division;
- host-side `floor`, `log2`, and `pow` if ALiBi parameters are prepared outside the kernel.

The exact HIP device function spellings should be recorded from real source rather than assumed from CUDA.

## Dot-product and reduction strategy space

A ROCm implementation could use one or more of:

- per-thread scalar accumulation followed by LDS reduction;
- wavefront shuffle reduction;
- architecture-specific matrix/MFMA instructions;
- library-backed matrix operations;
- fused streaming attention kernels.

This inventory deliberately does not choose among them. When code exists, document the actual strategy and any wavefront-size assumptions.

## Matrix/MFMA and architecture-specific acceleration

AMD hardware may expose matrix/fused-multiply acceleration through architecture-specific compiler intrinsics or ROCm libraries. These are relevant inventory categories because a performant ROCm xFormers implementation may not resemble the CUDA scalar/shared-memory kernel.

Record only when implemented:

- MFMA/WMMA-style intrinsics;
- rocWMMA usage;
- composable-kernel or library-backed attention pieces;
- architecture-specific tile sizes;
- LDS staging requirements.

## Mask, bias, and ALiBi behavior to inventory

The future ROCm implementation must explicitly record how it handles:

- additive masks;
- causal masking if supported;
- mask datatype/layout;
- broadcast across heads/batches;
- ALiBi slopes;
- logit soft-cap;
- attention sinks;
- fully masked rows;
- positive/negative infinity behavior;
- NaN propagation/normalization behavior.

No equivalence with the current CUDA implementation is asserted until this behavior exists and is tested.

## Softmax strategy to inventory

Possible native strategies include:

- materialized stable softmax;
- online softmax;
- block-local reduction;
- wavefront reduction;
- two-level wave + LDS reduction.

When the real ROCm path is implemented, record:

- accumulator precision;
- reduction width;
- treatment of masked `-inf` values;
- treatment of multiple `+inf` values;
- whether the score matrix is ever materialized;
- whether AV is fused with normalization.

## Error handling and runtime status

HIP runtime mechanisms include:

- function return values of type `hipError_t`;
- `hipGetLastError()`;
- `hipPeekAtLastError()`;
- `hipGetErrorString(...)`;
- explicit synchronization calls when asynchronous errors must be surfaced.

The actual API.cpp integration should document whether errors are converted to Common status, assertions, diagnostics, or backend failure codes.

## Device and architecture discovery

Relevant HIP runtime/device information includes:

- device count and selected device;
- architecture name/properties;
- maximum work-group/block dimensions;
- wavefront/warp size;
- LDS/shared-memory capacity;
- supported datatypes/instructions;
- total/free device memory;
- stream/device capability constraints.

Only properties actually used by the implementation should become translation requirements.

## Important architecture-specific facts to preserve

- AMD wavefront width is architecture-dependent.
- Source-level CUDA/HIP similarity does not imply identical scheduling or performance.
- Cross-lane mask width and ballot semantics can differ from CUDA assumptions.
- LDS capacity/banking and occupancy constraints differ from NVIDIA shared memory.
- Matrix-instruction availability differs substantially by GCN/CDNA/RDNA generation.
- Floating-point atomic support/performance varies by architecture and datatype.
- HIP does not provide CUDA dynamic parallelism equivalence for all targets; this is not currently required by xFormers but demonstrates why syntax similarity is insufficient.

## Current repository state

The ROCm xFormers `definition/` tree exists, but its current method files are structural placeholders and the ROCm `translation/` tree does not yet contain a functional xFormers implementation.

Therefore this inventory is intentionally a capability/primitive catalog, not a claim about code already running in this repository.

## Inventory gaps to fill from the first real ROCm implementation

- actual kernel launch form;
- exact stream/context object passed from ggml/API.cpp;
- Q/K/V layout requirements;
- dtype types and conversion intrinsics;
- reduction strategy;
- work-group size;
- wavefront assumptions;
- shared/LDS allocation;
- mask/bias/ALiBi handling;
- soft-cap handling;
- sink handling;
- output dtype;
- architecture gates;
- status/error propagation;
- runtime validation results by AMD architecture.

## Questions reserved for the later comparison phase

Do not answer these here.

- Which HIP calls are exact semantic matches to CUDA calls and which merely share names?
- Which concepts need architecture metadata in translation because wave size differs?
- Should reduction/shuffle behavior be translated as one logical primitive or left inside a fused backend operation?
- Which datatype conversions are native versus emulated on each AMD generation?
- Which ROCm execution choices are required for correctness versus chosen only for performance?
