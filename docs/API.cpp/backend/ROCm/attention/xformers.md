# ROCm — xFormers attention

## Status
Stub / not implemented.

## API role
The common ABI exposes xFormers-style attention without tying callers to CUDA or HIP symbols.

## Current behavior
The shared `ggml_xformers_attn_supported()` currently returns false for all backends, including ROCm. Calling `ggml_xformers_attn()` aborts because no native ROCm xFormers-style implementation is wired yet.

## Required work
Add a HIP/ROCm implementation behind the existing ABI, define gfx/tensor-layout capability rules, and enable the support probe only for validated configurations.

## Fallback
Use another supported attention implementation.

## Validation
No native ROCm runtime validation is applicable until the implementation exists.
