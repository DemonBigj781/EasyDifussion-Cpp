# FlashAttention

## Status
**Implemented — CUDA optimized compatibility path.**

JetPack 5.x uses CUDA 11.4 by default and supports the tracked upgrade range through CUDA 12.2. API.cpp uses ggml CUDA `fattn`; no JetPack-specific Flash kernel was found. Eligibility remains determined by fattn's device/tensor dispatch.

JetPack 5 hardware may be Xavier (SM72) or Orin (SM87), so runtime validation is device-specific. If fattn is ineligible, use a correct standard-attention fallback.