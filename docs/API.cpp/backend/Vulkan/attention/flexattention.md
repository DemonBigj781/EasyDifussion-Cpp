# FlexAttention — Vulkan

## Status

**Backend-neutral host implementation.**

The current FlexAttention token-selection policy executes in native host C++ and produces a host mask. No Vulkan compute-shader implementation currently performs the selection stage.

It can coexist with Vulkan model execution when the required attention data is available to the host. Device-native Vulkan support would require reductions, normalization, thresholding, adaptive pooling, mask expansion, and transfer/synchronization design.