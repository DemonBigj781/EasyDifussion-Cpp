# FlexAttention

## Status
**Implemented as host C++; not a Jetson CUDA kernel.**

The repository's visual-token-selection FlexAttention sums host float attention, normalizes, thresholds, adaptive-max-pools, and expands a host mask. JetPack 5/CUDA 12.0 adds no device implementation.

It can coexist with Xavier/Orin CUDA execution when the attention data is available on host; transfers and synchronization are the backend-specific concern.