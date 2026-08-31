# FlexAttention

## Status
**Implemented as backend-neutral host C++; not a CUDA kernel.**

This repository's FlexAttention is the high-resolution visual-token-selection policy. It sums host float attention across heads, normalizes, thresholds, adaptive-max-pools, then expands a host mask. CUDA 12.6 has no separate device implementation.

It can accompany CUDA inference when attention data is available on host; a device-native port would need reductions, normalization, thresholding, pooling, mask expansion, and transfer/synchronization handling.