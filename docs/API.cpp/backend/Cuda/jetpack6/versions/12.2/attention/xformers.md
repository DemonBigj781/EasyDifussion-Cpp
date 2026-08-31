# xFormers

## Status
**Stub / not implemented.**

The common xFormers API currently provides no usable CUDA xFormers execution path. JetPack 6 + CUDA 12.2 therefore remains unsupported even though Orin hardware is CUDA-capable.

Required work: implement CUDA execution, capability probing, dtype/shape rules, dispatch/fallback, and validation while preserving a compile-complete API surface.