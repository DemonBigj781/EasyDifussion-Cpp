# ROCm — FlashAttention compatibility

## Status
Compatibility / implemented through the shared HIP fattn path.

## API role
FlashAttention is an explicit compatibility contract rather than an automatically selected optimized-attention implementation.

## Implementation
`ggml_flash_compat_supported()` verifies that the target is an AMD/HIP device and delegates capability probing to `ggml_cuda_flash_attn_ext_supported()`. Execution is delegated to `ggml_cuda_flash_attn_ext()`; architecture-specific HIP kernel selection remains inside the existing `fattn` implementation.

## Hardware policy
Only AMD/HIP targets are accepted by this compatibility path. MFMA/WMMA/tile/vector selection remains internal to the HIP attention implementation.

## Fallback
If the compatibility probe returns false, callers must select another available attention path.

## Validation
Implementation is present. Representative gfx-family runtime validation is still required.
