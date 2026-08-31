# FlashAttention

## Status
**Implemented — CUDA optimized compatibility path.**

CUDA 12.2 is the default CUDA generation for JetPack 6.x. API.cpp uses ggml CUDA `fattn`; no JetPack-specific Flash kernel was found. Eligibility is still determined by fattn's device/tensor dispatch.

JetPack 6 targets Jetson Orin-class hardware; runtime validation remains required. If fattn is ineligible, dispatch must use a correct standard-attention fallback.