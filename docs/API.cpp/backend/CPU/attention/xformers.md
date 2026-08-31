# CPU — xFormers attention

## Status
Not implemented.

## Current architecture
The shared xFormers attention ABI currently exists as a backend-neutral contract for future optimized implementations, but the present execution function is tied to the CUDA/HIP backend context and the support probe returns false.

## CPU relationship
There is no CPU xFormers-style kernel or CPU adapter in the current tree.

## Fallback
CPU execution uses the normal supported attention implementation.

## Required work
A CPU implementation would need a CPU-specific execution path, capability rules, numerical tests, and a demonstrated reason to expose it as xFormers-compatible rather than using the existing CPU attention kernels.

## Validation
No CPU xFormers implementation is currently available to validate.
