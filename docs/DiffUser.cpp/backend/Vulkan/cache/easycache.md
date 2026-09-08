# EasyCache — Vulkan

## Status

**Native policy; host-resident cache state.**

EasyCache is backend-neutral native C++ and currently keeps its cache state in host containers. Vulkan model execution does not turn the cache into a Vulkan shader/device cache.

Only future Vulkan-specific transfer, synchronization, or device-resident acceleration belongs in the backend layer.