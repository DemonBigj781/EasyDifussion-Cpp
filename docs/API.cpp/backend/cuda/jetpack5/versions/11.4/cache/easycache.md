# EasyCache

## Status
**Implemented — backend-neutral host-side policy.**

EasyCache is restricted to supported DiT model types and defaults to a 0.2 reuse threshold when unspecified. Its cache policy/state is host-resident; JetPack 5/CUDA 11.4 has no EasyCache GPU kernel.

Jetson-specific API work is transfer/synchronization or future device-resident acceleration, not the generic cache algorithm.