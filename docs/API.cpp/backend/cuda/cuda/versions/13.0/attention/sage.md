# SageAttention

## Status
**Native CUDA implementation, hardware-gated.**

`ggml-cuda/sage/sage-attention-sm80.cu` accepts compute capability >= Ampere and < Hopper. Required tensors are Q=F32, K/V=F16, output=F32; head dimension must be 64 or 128; mask and sinks must be absent; max bias and logit softcap must both be zero; layout/contiguity checks must pass.

CUDA 13.0 does not override these source predicates. Unsupported devices/shapes must fall back to another attention implementation.