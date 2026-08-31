# TeaCache — Vulkan

## Status

**Native policy; host-resident cache state.**

TeaCache currently keeps previous input and residual data in host memory. It is a backend-neutral condition-boundary policy and has no Vulkan-specific cache shader/device-resident implementation.

Future Vulkan acceleration may optimize state placement or transfers without moving generic TeaCache policy into the backend.