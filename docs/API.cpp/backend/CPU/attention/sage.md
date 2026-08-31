# CPU — SageAttention

## Status
Not implemented as a CPU SageAttention backend.

## Current architecture
The current common SageAttention execution ABI is defined around the CUDA/HIP backend context and dispatches to the CUDA SM80 implementation when supported. There is no CPU SageAttention kernel or CPU-specific Sage execution adapter in the current tree.

## Fallback
CPU execution uses its existing attention implementations. This page should not imply SageAttention acceleration on CPU.

## Required work
A true CPU SageAttention implementation would need its own backend-neutral or CPU-specific execution ABI, capability rules, numerical validation, and performance justification.

## Validation
No CPU SageAttention implementation is currently available to validate.
