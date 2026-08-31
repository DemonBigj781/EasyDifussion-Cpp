# CUDA — FlashAttention compatibility

## Status
Compatibility / implemented for supported Volta-or-newer NVIDIA targets.

## API role
FlashAttention is retained as an explicit compatibility contract and is not part of automatic optimized-attention selection. Callers use `ggml_flash_compat_supported()` and `ggml_flash_compat()`.

## Implementation
The common API delegates execution to the existing CUDA `fattn` implementation through `ggml_cuda_flash_attn_ext_supported()` and `ggml_cuda_flash_attn_ext()`.

## Hardware policy
The generic CUDA compatibility path accepts NVIDIA compute capability Volta or newer. Pascal is intentionally excluded from this generic path and is reserved for a separate compatibility implementation.

## Fallback
If the compatibility probe returns false, the caller must use another supported attention path rather than invoking the compatibility function.

## Validation
Implementation is present. Runtime behavior still requires architecture-specific validation on representative NVIDIA hardware.
