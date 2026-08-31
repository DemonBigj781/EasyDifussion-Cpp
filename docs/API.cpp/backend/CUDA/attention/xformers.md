# CUDA — xFormers attention

## Status
Stub / not implemented.

## API role
The common attention ABI exposes `ggml_xformers_attn_supported()` and `ggml_xformers_attn()` so a CUDA implementation can be added without changing higher-level callers.

## Current behavior
`ggml_xformers_attn_supported()` returns false for all devices. Invoking `ggml_xformers_attn()` aborts because no native xFormers-style kernel is wired into this ggml tree yet.

## Required work
Add a native CUDA implementation behind the existing common ABI, define capability constraints, then enable the support probe only for validated devices and tensor layouts.

## Fallback
Use another supported attention implementation when xFormers support is unavailable.

## Validation
No runtime validation is applicable until an implementation is wired.
