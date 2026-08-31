# xFormers

## Status
**Stub / not implemented.**

The common xFormers API surface is present, but the current implementation does not expose a usable CUDA xFormers kernel and reports unsupported. CUDA 12.2 does not change this.

## Required work
Implement CUDA execution, capability probing, dtype/shape rules, dispatch/fallback behavior, and correctness/performance validation. The symbol surface should remain compile-complete while runtime capability is false.