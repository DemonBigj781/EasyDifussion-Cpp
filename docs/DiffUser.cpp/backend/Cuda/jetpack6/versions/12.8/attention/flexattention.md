# FlexAttention

## Status
**Implemented as host C++; not an Orin CUDA kernel.**

The repository's FlexAttention is a visual-token-selection policy that sums host float attention, normalizes, thresholds, adaptive-max-pools, and expands a host mask. JetPack 6/CUDA 12.8 adds no device implementation.

It can coexist with Orin CUDA inference when attention data is available on host; transfers and synchronization are the backend-specific concern.