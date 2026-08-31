# SageAttention — CUDA

## Status

**Native** on supported NVIDIA targets.

## Implementation

The common optimized-attention API routes SageAttention through `ggml_sage_attn_supported()` and `ggml_sage_attn()`. CUDA currently dispatches to the SM80 implementation through `sage-attention-sm80.cuh` / `sage-attention-sm80.cu`.

## Hardware targets

The currently wired native implementation is SM80-oriented. Hardware outside the implementation's supported capability check must not be assumed compatible.

## Precision / data types

See the SM80 implementation and its capability check for the authoritative tensor-layout, head-size, and data-type restrictions. These restrictions should remain hidden behind the common API rather than leaking into higher-level application code.

## Compile-time requirements

- CUDA backend enabled.
- SageAttention sources included in the CUDA build.

## Runtime capability check

Call `ggml_sage_attn_supported(device, dst)` before dispatch. The CUDA path delegates the decision to `ggml_cuda_sage_attn_sm80_supported(device, dst)`.

## Fallback

If SageAttention is not supported for the selected device/tensor, the shared attention dispatcher must select another supported attention implementation rather than calling the Sage kernel.

## Known limitations

- Native path currently centers on the SM80 implementation.
- Additional NVIDIA architectures may require separate implementations or explicit validation.

## Validation

The common CUDA/Sage dispatch has compile-time integration in the Theory branch. Hardware runtime validation should be recorded here as targets are tested.

## Relevant source paths

- `source/sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/sage/sage-attention.cuh`
- `source/sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/sage/sage-attention.cu`
- `source/sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/sage/sage-attention-sm80.cuh`
- `source/sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/sage/sage-attention-sm80.cu`
