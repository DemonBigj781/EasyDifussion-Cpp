# FlashAttention

## Status
**Implemented — CUDA optimized compatibility path.**

CUDA 12.2 is compatible with JetPack 5.x through NVIDIA's CUDA upgrade path and is the final compatible version in the tracked JetPack 5 range. API.cpp uses ggml CUDA `fattn`; no JetPack-specific Flash kernel was found. Eligibility remains determined by fattn's device/tensor dispatch.

Runtime validation remains device-specific; if fattn is ineligible, use a correct standard-attention fallback.