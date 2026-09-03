# CUDA native-call inventory for xFormers

Status: raw inventory. Do not derive normalized translation names from this file yet.

## Execution and launch

- `__global__` kernel entry points.
- Triple-chevron launch syntax `kernel<<<grid, block, shared_mem, stream>>>(...)`.
- `dim3` for grid/block geometry.
- `cudaStream_t` for ordered asynchronous execution.
- `cudaGetLastError()` for post-launch error observation.

## Thread/block identity

- `threadIdx.x` — lane/thread position inside the CUDA block.
- `blockIdx.x` — block position inside the launch grid.
- `blockDim` / `gridDim` are relevant general CUDA geometry primitives even where the current xFormers kernel uses constants and explicit launch dimensions.

## Synchronization and shared storage

- `__shared__` block-local storage.
- `__syncthreads()` block-wide barrier plus memory visibility for participating threads.
- `__syncwarp(mask)` is a CUDA warp-level synchronization primitive relevant if later implementations replace block reductions with warp/subgroup operations.

## Arithmetic / math used by current implementation

- `isnan`
- `fmaxf`
- `expf`
- `tanhf`
- `floor`
- `log2`
- `pow`
- CUDA infinity constants such as `CUDART_INF_F`.

## Scalar and dtype conversion

- `half`
- `nv_bfloat16`
- `__half2float(...)`
- `__uint_as_float(...)`
- explicit `reinterpret_cast` loads/stores for F32/F16/BF16 data.

## Current xFormers-specific execution pattern

The CUDA path currently launches one block per logical attention row and uses 256 threads per block. Q·K is accumulated cooperatively into block-local shared memory, reduced using repeated `__syncthreads()`, and fed into an online-softmax state. The output numerator is accumulated as K/V positions stream through the kernel, so the full QK score matrix is not materialized.

The current implementation also handles:

- Q/K/V head and batch remapping;
- additive F16 mask values;
- ALiBi slope contribution;
- logit soft-cap behavior;
- attention sinks that affect the denominator without a V row;
- F32/F16/BF16 Q/K/V dispatch;
- F32 output;
- architecture gating at Pascal or newer NVIDIA CUDA devices;
- layout and shape validation before launch.

## Current repository symbols to preserve in later comparison

- `ggml_cuda_xformers_attn_supported(...)`
- `ggml_cuda_xformers_attn(...)`
- `xformers_mea_forward_kernel(...)`
- `xformers_online_softmax_update(...)`
- `xformers_launch(...)`
- `xformers_launch_v(...)`
- `xformers_launch_kv(...)`

## Questions for the later comparison phase

Do not answer these here; they belong in the cross-backend comparison.

- Which CUDA concepts have exact HIP equivalents?
- Which block/warp assumptions depend on NVIDIA warp width or independent-thread scheduling?
- Which dtype conversions have exact native analogues versus software reconstruction?
- Which execution/error objects should remain backend-specific inside translation?
- Which operations are semantic requirements of xFormers versus implementation choices of this CUDA kernel?
