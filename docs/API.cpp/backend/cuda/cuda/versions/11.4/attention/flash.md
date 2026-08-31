# FlashAttention

## Status
**Implemented — CUDA optimized compatibility path.**

API.cpp uses ggml CUDA `fattn` for the explicit FlashAttention compatibility surface. No Flash-specific CUDA-toolkit version gate was found in the repository; eligibility is determined by the CUDA fattn dispatcher, tensor shape/type, and device support.

## Fallback
If fattn is ineligible, dispatch must use a correct non-Flash attention path rather than treating generic attention as FlashAttention.

## Version note
This source-level status is unchanged for CUDA 11.4; compile/runtime validation is still required for the selected GPU architecture.