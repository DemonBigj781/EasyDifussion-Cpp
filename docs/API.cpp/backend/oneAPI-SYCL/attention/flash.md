# FlashAttention — oneAPI / SYCL

## Status

**Not integrated into the common Flash compatibility API.**

The repository contains a substantial `ggml-sycl` backend, but the API.cpp/common compatibility surface currently exposes Flash compatibility through CPU and CUDA/HIP paths. No SYCL implementation is currently wired into `ggml_flash_compat_supported()` / `ggml_flash_compat()`.

This does not mean SYCL cannot execute attention operations; it means the explicit API.cpp FlashAttention compatibility contract has not yet been implemented for SYCL.

## Work required

Add a SYCL-side compatibility implementation or adapter, capability probing, datatype/head-size restrictions, fallback policy, and runtime validation before marking this Native or Compatibility.