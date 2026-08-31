# Common attention

This directory owns backend-neutral attention contracts and host implementations.

- `ggml-attention-common.h` defines optimized-attention identifiers and the CPU FlashAttention compatibility adapter.
- `flex_attention.hpp` and `flex_attention.cpp` implement the host-side FlexAttention token-selection API.

Device-specific attention kernels belong under their corresponding backend directory.
