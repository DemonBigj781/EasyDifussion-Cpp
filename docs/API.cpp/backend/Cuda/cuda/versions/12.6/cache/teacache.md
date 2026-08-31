# TeaCache

## Status
**Implemented — backend-neutral C++; host-resident state.**

TeaCache is enabled for supported DiT model types, tracks prior diffusion/residual state on host, and applies model-specific reuse/rescaling policy. The runtime includes model-specific coefficients for LTX-Video and Wan variants. There is no CUDA-specific TeaCache kernel in CUDA 12.6.

CUDA-specific work is transfer/synchronization or future device-resident acceleration, not moving generic TeaCache policy into the backend.