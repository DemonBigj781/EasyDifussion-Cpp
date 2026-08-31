# CPU — FlashAttention compatibility

## Status
Compatibility / implemented.

## API role
CPU exposes FlashAttention-compatible behavior through the common compatibility concept rather than claiming GPU FlashAttention performance.

## Implementation
`ggml_cpu_flash_compat_supported()` accepts tensors using `GGML_OP_FLASH_ATTN_EXT`. `ggml_cpu_flash_compat()` delegates to the existing `ggml_compute_forward_flash_attn_ext()` implementation, which remains the source of truth.

## Performance note
This is API/correctness retention. It should not be documented or benchmarked as equivalent to CUDA/HIP FlashAttention kernels.

## Fallback
Unsupported operations continue through the normal CPU graph paths.

## Validation
Compatibility adapter is present and shares the existing CPU implementation.
