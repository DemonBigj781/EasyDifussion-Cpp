# FlashAttention — Vulkan

## Status

**Not integrated into the common Flash compatibility API.**

The Vulkan backend is present, but the current explicit Flash compatibility API is implemented for CPU and CUDA/HIP paths. No Vulkan adapter is wired into that contract.

Generic Vulkan attention execution must not be labeled FlashAttention compatibility until its semantics, supported shapes/types, dispatch, and validation are explicitly connected to API.cpp.