# CUDA native-call inventory for xFormers

Status: raw inventory. Do not derive normalized translation names from this file yet.

Purpose: record CUDA-native execution, memory, synchronization, datatype, math, launch, validation, and error primitives that may be required by an xFormers-compatible implementation. This file describes CUDA as CUDA; it does not define the future common translation vocabulary.

## Execution and compilation model

- `__global__` marks device kernel entry points callable from host code.
- `__device__` marks device-only functions.
- `__host__` may coexist with `__device__` for dual-target helpers when useful.
- `__forceinline__` is used for device helpers where inlining is explicitly requested.
- Template specialization is used by the current implementation to select F32/F16/BF16 load behavior and Q/K/V launch combinations.
- Preprocessor architecture/backend gates are used to exclude unsupported compilation targets.

## Kernel launch and execution objects

- Triple-chevron launch syntax `kernel<<<grid, block, shared_mem, stream>>>(...)`.
- `dim3` describes grid/block geometry.
- `cudaStream_t` represents an ordered asynchronous execution stream.
- `ctx.stream()` supplies the stream used by the current ggml CUDA backend integration.
- Static shared-memory size may be declared in the kernel; dynamic shared-memory bytes may also be supplied as the third launch parameter when needed by future kernels.
- Kernel launch is asynchronous with respect to the host unless later synchronization or dependency handling requires otherwise.

## Thread, block, grid, and warp identity

- `threadIdx.x/y/z` — thread coordinates inside a block.
- `blockIdx.x/y/z` — block coordinates inside a grid.
- `blockDim.x/y/z` — dimensions of the executing block.
- `gridDim.x/y/z` — dimensions of the executing grid.
- `warpSize` — CUDA warp width exposed to device code.
- Current xFormers code uses `threadIdx.x` and `blockIdx.x` directly and launches one block per logical attention row.

## Block-local/shared storage

- `__shared__` declares block-local shared memory.
- Current CUDA xFormers uses a shared `float reduction[256]` plus shared online-softmax scalars (`alpha`, `beta`, `running_max`, `running_sum`).
- Shared-memory lifetime is one block execution.
- Shared memory is explicitly synchronized before producer/consumer phases in the current reduction loop.

## Synchronization

- `__syncthreads()` — block-wide barrier and ordering point for participating threads; used repeatedly by the current Q·K reduction and online-softmax/output update loop.
- `__syncwarp(mask)` — warp-level synchronization primitive relevant to future warp-specialized implementations.
- Warp vote/shuffle primitives are not used by the current xFormers kernel but belong in the CUDA inventory because a later optimized implementation may use them instead of shared-memory block reductions.

Candidate CUDA warp primitives to inventory if/when used:

- `__shfl_sync`
- `__shfl_up_sync`
- `__shfl_down_sync`
- `__shfl_xor_sync`
- `__ballot_sync`
- `__any_sync`
- `__all_sync`
- `__activemask`

## Memory ordering and fences

Not currently required by the inspected xFormers kernel beyond block barriers, but CUDA provides:

- `__threadfence_block()`
- `__threadfence()`
- `__threadfence_system()`

These must remain separate inventory items because a future backend implementation may need ordering that is not equivalent to a block barrier.

## Global-memory addressing and layout

The current CUDA path operates on raw tensor byte pointers and explicit byte strides:

- `const char*` / `char*` base pointers;
- `reinterpret_cast` to typed row elements;
- `ne[]` dimensions for logical tensor extents;
- `nb[]` byte strides for physical tensor layout;
- manual batch/head/token addressing;
- output addressing through destination byte strides.

The implementation does not assume all higher dimensions are tightly packed. It does require a supported contiguous element layout at dimension 0.

## Q/K/V head and batch mapping

Current device-side indexing derives:

- logical batch from block row;
- query head from row position;
- query token position from row position;
- K head via grouped-head division;
- V head via grouped-head division;
- K batch and V batch via batch-ratio division.

The support check requires query head counts and query batch counts to divide cleanly by K/V counts.

## Dot-product behavior

Current Q·K execution:

- each block processes one query-row/head/batch combination;
- each thread accumulates a strided subset of the head dimension;
- partial dot products are written to shared memory;
- a power-of-two block reduction combines partial sums;
- the reduced value is scaled before mask/bias/softmax processing.

Current code uses scalar float accumulation even when Q/K inputs are F16 or BF16.

## Scaling and logit soft-cap behavior

The current implementation reads operation parameters from `dst->op_params` using `std::memcpy`:

- scale;
- maximum bias;
- logit soft-cap;
- requested precision.

Behavior includes:

- direct multiply by scale when no soft-cap is configured;
- scale divided by soft-cap before the dot-product result is transformed when soft-cap is active;
- `tanhf`-based logit soft-cap transformation;
- rejection of non-finite scale/bias/soft-cap values;
- rejection of negative soft-cap values;
- current acceptance of default or F32 precision modes.

## Mask and bias behavior

Current CUDA xFormers supports an optional F16 mask tensor.

Native operations involved include:

- byte-stride mask addressing;
- F16-to-F32 mask conversion via `__half2float`;
- additive application to the score;
- ALiBi slope multiplication before addition.

Mask head and mask batch dimensions may be broadcast/remapped using modulo indexing, subject to pre-launch divisibility checks.

## ALiBi behavior

The current implementation computes host-side slope parameters from:

- `max_bias`;
- query head count;
- `floor(log2(...))`;
- `pow(...)`.

Device code then calls the existing `get_alibi_slope(...)` helper per query head and applies the resulting slope to mask values.

This is an implementation fact to preserve during comparison; it is not yet a proposed common API.

## Online softmax behavior

The CUDA path does not materialize a complete score row. Instead it maintains an online state:

- `running_max`;
- `running_sum`;
- `alpha` to rescale the existing output numerator;
- `beta` as the weight of the newly observed key/value position.

Native math/control behavior used by the update includes:

- `isnan`
- comparisons against `+/-CUDART_INF_F`
- `fmaxf`
- `expf`

Special cases are explicitly handled for:

- NaN logits;
- negative infinity / fully masked contributions;
- first valid value;
- multiple positive-infinity logits;
- finite logits following positive infinity.

## Attention-sink behavior

An optional F32 sink tensor may add one denominator-only softmax contribution per query head.

The sink:

- participates in online-softmax state update;
- has no corresponding V row;
- therefore rescales the accumulated numerator through `alpha` without adding a new V contribution.

## AV/output accumulation

For each K/V position, each thread owns up to two output dimensions in the current kernel.

The update is:

`out_acc = out_acc * alpha + beta * V`

After all K/V positions and optional sinks are processed, the accumulated numerator is multiplied by `1 / running_sum` and stored as F32 output.

## Scalar types and conversion primitives

Current supported Q/K/V input types:

- `float` / GGML F32;
- `half` / GGML F16;
- `nv_bfloat16` / GGML BF16.

Current conversion/loading behavior:

- direct cast/load for F32;
- `__half2float(...)` for F16;
- BF16 reconstruction using a 16-bit load, shift, and `__uint_as_float(...)`;
- `reinterpret_cast` for typed reads and writes.

Current output type is F32.

Potential CUDA datatype primitives worth retaining in inventory if later code uses them:

- `__float2half*` conversion family;
- `__float2bfloat16*` / BF16 conversion family;
- vectorized half/bfloat forms such as `half2`/BF16 pair types;
- packed/vector loads when layout permits.

They are not claimed as requirements of the current implementation.

## Math primitives used or structurally relevant

Current host/device implementation uses:

- `isnan`
- `fmaxf`
- `expf`
- `tanhf`
- `std::isfinite`
- `std::floor`
- `std::log2`
- `std::pow`
- `std::numeric_limits<int32_t>::max()`
- CUDA infinity constant `CUDART_INF_F`.

## Atomics

The current CUDA xFormers kernel does not use atomics because each block owns its output row.

CUDA atomics remain relevant inventory categories for alternative decompositions:

- `atomicAdd`
- `atomicCAS`
- `atomicExch`
- integer/bitwise/min/max atomic families

Do not infer that a future normalized xFormers API requires atomics merely because CUDA exposes them.

## Capability and architecture gating

Current support logic checks:

- destination operation is `GGML_OP_FLASH_ATTN_EXT`;
- selected device index is valid;
- target is NVIDIA CUDA rather than HIP/MUSA through this source path;
- compute capability is Pascal or newer;
- Q/K/V element types are supported;
- dimension-0 byte stride equals native element size;
- destination type/layout is F32-compatible;
- Q/K head dimensions match where required;
- V dimension is positive and no greater than the current 512-value kernel limit;
- K/V token counts agree;
- head/batch ratios divide exactly;
- destination dimensions correspond to V dimension, Q heads, Q length, Q batches;
- optional mask type/layout/extents are valid;
- optional sink type/layout/head count are valid;
- operation parameters are finite and supported;
- row count fits the launch's 32-bit row representation.

## Error and assertion behavior

Current mechanisms include:

- support predicate returning `false` before dispatch;
- `GGML_ASSERT(...)` when the execution function is called;
- `GGML_ABORT(...)` for unreachable/unsupported dtype dispatch cases;
- `cudaGetLastError()` wrapped through `CUDA_CHECK(...)` after launch.

Host/device synchronization is not performed merely to turn every asynchronous runtime failure into an immediate host error; that distinction must be preserved later.

## Current xFormers-specific execution pattern

The CUDA path is a fused streaming implementation:

1. identify one logical query row per block;
2. stream K/V positions;
3. cooperatively calculate Q·K;
4. apply scale, soft-cap, mask, and ALiBi;
5. update online softmax;
6. update the V-weighted numerator;
7. optionally incorporate attention sinks;
8. normalize and write F32 output.

The full QK score matrix is not materialized.

## Current repository symbols to preserve in later comparison

- `ggml_cuda_xformers_attn_supported(...)`
- `ggml_cuda_xformers_attn(...)`
- `xformers_forward_params`
- `xformers_load<T>(...)`
- `xformers_mea_forward_kernel(...)`
- `xformers_online_softmax_update(...)`
- `xformers_supported_type(...)`
- `xformers_type_size(...)`
- `xformers_valid_row_layout(...)`
- `xformers_read_params(...)`
- `xformers_launch(...)`
- `xformers_launch_v(...)`
- `xformers_launch_kv(...)`

## Inventory gaps to fill from future CUDA implementations

Record these only when real code uses them:

- warp/subgroup optimized reductions;
- cooperative groups;
- tensor-core/MMA paths;
- asynchronous global-to-shared copies;
- vectorized loads/stores;
- explicit events;
- stream waits/dependencies;
- graph capture/launch;
- occupancy-driven launch selection;
- architecture-specialized kernels;
- quantized Q/K/V paths.

## Questions reserved for the later comparison phase

Do not answer these here.

- Which CUDA concepts have exact HIP equivalents?
- Which block/warp assumptions depend on NVIDIA warp width or independent-thread scheduling?
- Which dtype conversions have exact native analogues versus software reconstruction?
- Which execution/error objects should remain backend-specific inside translation?
- Which operations are semantic requirements of xFormers versus choices of this CUDA kernel?
- Should a normalized call expose fused online attention, staged QKT/mask/softmax/AV, or capabilities that allow either?
