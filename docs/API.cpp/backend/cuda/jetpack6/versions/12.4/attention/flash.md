# FlashAttention

## Status
**Implemented — CUDA optimized compatibility path.**

CUDA 12.4 is compatible with JetPack 6.x through NVIDIA's CUDA upgrade support. API.cpp uses ggml CUDA `fattn`; no JetPack-specific Flash kernel was found. Eligibility is still determined by fattn's device/tensor dispatch.

JetPack 6 targets Jetson Orin-class hardware; runtime validation remains required. If fattn is ineligible, dispatch must use a correct standard-attention fallback.