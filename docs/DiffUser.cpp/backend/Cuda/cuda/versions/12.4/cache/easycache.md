# EasyCache

## Status
**Implemented — backend-neutral C++; host-resident state.**

EasyCache is enabled only for supported DiT model types and defaults to a 0.2 reuse threshold when unspecified. Its policy/state is not a CUDA kernel or device-resident cache. CUDA 12.4 therefore does not change the algorithm.

CUDA-specific API work should be limited to transfers, synchronization, or a future device-resident acceleration path; generic EasyCache policy remains outside backend code.