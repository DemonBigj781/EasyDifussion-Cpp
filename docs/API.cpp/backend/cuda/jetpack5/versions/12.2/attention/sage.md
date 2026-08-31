# SageAttention

## Status
**Native CUDA implementation; Jetson hardware decides availability.**

The Sage predicate accepts compute capability >= 8.0 and < 9.0. Therefore Jetson Xavier SM72 is unsupported, while Jetson Orin SM87 passes the architecture gate. It additionally requires Q=F32, K/V=F16, output=F32, head dimension 64 or 128, no mask/sinks, zero max-bias and logit-softcap, and valid layout/contiguity.

CUDA 12.2 is supported by JetPack 5; unsupported cases must fall back.