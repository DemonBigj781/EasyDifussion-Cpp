# FlashAttention

## Status
**Runtime tested — normalized CUDA route; optimized compatibility path retained.**

The normalized API.cpp route uses the registered Common CUDA translation in
`features/attention/flash/cuda/translation/gpu/forward.cu`. Its fused
online-softmax implementation passed deterministic numerical tests and Compute
Sanitizer memcheck on an RTX 3060 (`sm_86`, driver 580.94.18) with CUDA 12.4.

Stable-diffusion.cpp still uses the separate optimized GGML CUDA `fattn`
compatibility path. No Flash-specific CUDA-toolkit version gate was found for
that path; eligibility remains determined by its dispatcher, tensor shape/type,
and device support. It does not yet establish application-to-Common routing.

## Fallback
If fattn is ineligible, dispatch must use a correct non-Flash attention path rather than treating generic attention as FlashAttention.

## Version note
The Common CUDA translation requires Pascal (`sm_60`) or newer. The local
Nouveau-driven Quadro K2000 (`sm_30`) is not a CUDA runtime target; validation
used only the RTX 3060.
