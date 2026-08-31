# Vulkan backend

## Backend status

**Present.**

The stable-diffusion.cpp tree contains a `ggml-vulkan` backend and Vulkan compute shaders. It is therefore a real API.cpp backend target.

The presence of the generic Vulkan backend does not by itself establish native support for each optimized attention family. Attention pages below distinguish generic backend execution from explicit API.cpp attention integration.

EasyCache, TeaCache, and the current FlexAttention token-selection policy are host-side/backend-neutral unless a future Vulkan-specific acceleration path is added.