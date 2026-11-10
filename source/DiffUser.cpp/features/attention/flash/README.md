# FlashAttention

FlashAttention has one backend-neutral Common request and registered CPU and
CUDA translations.

## Common CUDA route

`cuda/translation/gpu/forward.cu` is the authoritative normalized CUDA
translation. It converts Common requests and invokes
`cuda/definition/gpu/forward.cu`, which owns CUDA validation, runtime calls,
and the fused online-softmax kernel without materializing the QK score matrix.

The current CUDA Common route supports:

- F32, F16, and BF16 Q/K/V inputs with F32 accumulation and output;
- additive F32, F16, or BF16 masks with head/batch broadcasting;
- GGML-style max-bias/ALiBi mask scaling and logit soft-capping;
- grouped-query and multi-query head mappings;
- padded and safely permuted non-overlapping byte-stride layouts;
- an optional opaque CUDA stream and synchronous or asynchronous return;
- Pascal-or-newer CUDA devices, with a maximum value dimension of 512.

`source/API.test/Feature/Attention/Flash/Cuda/Main.cu` calls Common exclusively.
After the ownership relocation it passed numerical runtime and Compute
Sanitizer memcheck on an RTX 3060 (`sm_86`, driver 580.94.18, CUDA 12.4).

`source/API.test/MultiTest/flash/cycle/flash-api-roundtrip-cuda-test.cu`
additionally completed 32 Common model-load, device-byte return, Flash forward,
result-return, and Common model-unload cycles under memcheck. It is an API
roundtrip test, not an image-generation end-to-end claim.

The historical stable-diffusion.cpp CUDA adapter converted supported
`GGML_OP_FLASH_ATTN_EXT` tensors and parameters into this Common request when
`SD_CUDA_FLASH_COMMON=1`. Before the adapter was removed by the `d883d6e`
isolation change, the image-generation MultiTest completed a 512x512 one-step
SD 1.5 generation in 26.7 seconds with 40 observed Common launches. This is
historical evidence, not current end-to-end maturity.

## Compatibility paths

The CPU translation maps the normalized request to the native CPU definition,
which owns validation and FlashAttention execution. It currently permits one
host dispatch thread until native thread-pool scheduling is represented in the
Common contract.

The corrected CPU/CUDA ownership split has been compiled and runtime-revalidated
through Common. The CPU self-contained route, CUDA numerical route, and CUDA
roundtrip route all pass.

The CUDA `fattn*.{cu,cuh}` files and their template instances in the upstream
stable-diffusion.cpp snapshot remain an engine-owned optimized path. They are
separate from DiffUser's `forward.cu`; their execution does not establish
Common validation. The retained image harness still expects the removed Common
launch-counter symbol and therefore cannot produce a false passing integration
result.

The Nouveau-driven Quadro K2000 is not a CUDA target for this route. Its
Kepler `sm_30` architecture also predates the route's Pascal minimum.
