# FlashAttention

## Status
**Implemented — CUDA optimized compatibility path.**

CUDA 12.0 is compatible with JetPack 5.x through NVIDIA's CUDA upgrade path. API.cpp uses ggml CUDA `fattn`; no JetPack-specific Flash kernel was found. Eligibility remains determined by fattn's device/tensor dispatch.

JetPack 5 hardware may be Xavier (SM72) or Orin (SM87), so runtime validation is device-specific. If fattn is ineligible, use a correct standard-attention fallback.