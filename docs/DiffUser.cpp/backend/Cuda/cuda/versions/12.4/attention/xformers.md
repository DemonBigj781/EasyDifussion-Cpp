# xFormers

## Status
**End-to-end tested — single Common-routed CUDA forward path.**

The stable-diffusion.cpp application adapter converts `GGML_OP_FLASH_ATTN_EXT`
tensors into the backend-neutral Common request. Common dispatches to the
registered CUDA translation, which converts to the CUDA definition, validates
the active device and pointer address spaces, and launches the fused online-
softmax kernel. The route supports strided F32/F16/BF16 Q/K/V, F32 output,
supported additive-mask dtypes and broadcasting, causal masking, explicit or
maximum-bias ALiBi, logit soft-cap, attention sinks, GQA/MQA, head/batch
broadcasting, an opaque CUDA stream, and value dimensions through 512.

## Validation
Revalidated on 2026-09-05 using an RTX 3060 (`sm_86`), NVIDIA driver
580.94.18 (CUDA 13.0 runtime compatibility), and CUDA Toolkit 12.4.131: the
exact `xformers-load-unload-cuda-test` executable passes Common
analytical/conformance and rejection cases, 32 custom model
load/forward/unload cycles, and Compute Sanitizer memcheck with zero errors. The
standalone CUDA generator also loaded SD 1.5 and produced a 512x512 image; it
requires a positive native xFormers launch count so a silent ggml fallback
cannot pass that test.
