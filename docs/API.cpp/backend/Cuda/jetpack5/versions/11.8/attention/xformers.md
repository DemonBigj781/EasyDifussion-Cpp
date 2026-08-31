# xFormers

## Status
**Stub / not implemented.**

No usable CUDA xFormers execution path is currently provided by the common API, so JetPack 5 + CUDA 11.8 reports unsupported regardless of Xavier/Orin hardware.

Required work: implement CUDA execution, capability probing, dtype/shape rules, dispatch/fallback, and validation while keeping the API symbols compile-complete.