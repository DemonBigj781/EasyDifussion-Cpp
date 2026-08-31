# TeaCache

## Status
**Implemented — backend-neutral host-side policy.**

TeaCache is restricted to supported DiT model types, keeps reusable diffusion/residual state on host, and includes model-specific policy/coefficients such as LTX-Video and Wan variants. JetPack 6/CUDA 12.6 has no TeaCache GPU kernel.

Orin-specific work is transfer/synchronization or future device-resident acceleration.