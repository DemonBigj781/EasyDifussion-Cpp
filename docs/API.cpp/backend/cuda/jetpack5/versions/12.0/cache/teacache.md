# TeaCache

## Status
**Implemented — backend-neutral host-side policy.**

TeaCache is restricted to supported DiT model types, keeps reusable diffusion/residual state on host, and includes model-specific policy/coefficients such as LTX-Video and Wan variants. JetPack 5/CUDA 12.0 has no TeaCache GPU kernel.

Jetson-specific work is transfer/synchronization or future device-resident acceleration.