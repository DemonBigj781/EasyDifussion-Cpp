# FlashAttention

FlashAttention has one backend-neutral Common request and registered CPU and
CUDA translations.

## Common CUDA route

`cuda/translation/gpu/forward.cu` is the authoritative normalized CUDA
translation. It launches a fused, online-softmax forward kernel without
materializing the QK score matrix. The CUDA definition validates native
capabilities, tensor layouts, shapes, and launch bounds before translation.

The current CUDA Common route supports:

- F32, F16, and BF16 Q/K/V inputs with F32 accumulation and output;
- additive F32, F16, or BF16 masks with head/batch broadcasting;
- GGML-style max-bias/ALiBi mask scaling and logit soft-capping;
- grouped-query and multi-query head mappings;
- padded, non-overlapping byte-stride layouts;
- an optional opaque CUDA stream and synchronous or asynchronous return;
- Pascal-or-newer CUDA devices, with a maximum value dimension of 512.

`source/API.test/Feature/Attention/Flash/Cuda/Main.cu` calls Common exclusively.
It has numerical runtime and Compute Sanitizer memcheck evidence on an RTX 3060
(`sm_86`, driver 580.94.18, CUDA 12.4).

## Compatibility paths

The CPU translation maps the normalized request to the existing GGML
`FLASH_ATTN_EXT` implementation. It currently permits one host dispatch thread
until native thread-pool scheduling is represented in the Common contract.

The CUDA `fattn*.{cu,cuh}` files and their template instances remain the live
optimized GGML compatibility implementation selected by stable-diffusion.cpp.
They are separate from `forward.cu`: the application adapter has not yet been
migrated to the normalized Common route. Their presence therefore does not
establish Common or end-to-end validation.

The Nouveau-driven Quadro K2000 is not a CUDA target for this route. Its
Kepler `sm_30` architecture also predates the route's Pascal minimum.
