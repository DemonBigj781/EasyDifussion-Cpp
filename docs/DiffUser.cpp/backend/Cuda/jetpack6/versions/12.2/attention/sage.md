# SageAttention

## Status
**Native CUDA implementation and architecture-compatible with Jetson Orin.**

The Sage predicate accepts compute capability >= 8.0 and < 9.0. Jetson Orin is SM87, so it passes the architecture gate. Additional requirements are Q=F32, K/V=F16, output=F32, head dimension 64 or 128, no mask/sinks, zero max-bias and logit-softcap, and valid layout/contiguity.

CUDA 12.2 is JetPack 6's default; predicate failures must fall back to another attention path.