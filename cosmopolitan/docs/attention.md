# Attention in the all-in-one Cosmopolitan port

Status: research and integration requirements, checked on 2026-10-06.
This document does not claim that an attention kernel has been ported or
benchmarked merely because its upstream source or a Theory document exists.

## Existing Theory work

The historical reference is
[Theory at aa54da093aeeb9fa9ce34163f73d5797c31c316d](https://github.com/DemonBigj781/EasyDifussion-Cpp/tree/aa54da093aeeb9fa9ce34163f73d5797c31c316d).
Relevant documents include:

- [CPU attention audit](https://github.com/DemonBigj781/EasyDifussion-Cpp/blob/aa54da093aeeb9fa9ce34163f73d5797c31c316d/Audit/2026-09-08/CPU_ATTENTION_AUDIT.md).
- [Inference primitive audit](https://github.com/DemonBigj781/EasyDifussion-Cpp/blob/aa54da093aeeb9fa9ce34163f73d5797c31c316d/Audit/2026-09-08/INFERENCE_PRIMITIVE_AUDIT.md).
- [CUDA 12.8 FlashAttention notes](https://github.com/DemonBigj781/EasyDifussion-Cpp/blob/aa54da093aeeb9fa9ce34163f73d5797c31c316d/docs/DiffUser.cpp/backend/Cuda/cuda/versions/12.8/attention/flash.md).
- [CPU SageAttention notes](https://github.com/DemonBigj781/EasyDifussion-Cpp/blob/aa54da093aeeb9fa9ce34163f73d5797c31c316d/docs/DiffUser.cpp/backend/CPU/attention/sage.md).

The CPU audit distinguishes implemented execution from a capability declaration
or an empty directory. Its historical xFormers and Flash test results belong to
that branch and build environment. They are not Windows/Linux APE evidence.
The audit also records unfinished Flex, Split and Sage CPU work. Preserve those
distinctions when carrying documentation or code into this branch.

| Family | Meaning for this port | Required evidence |
| --- | --- | --- |
| Ordinary dense attention | Correct reference and fallback for each supported model | Shapes, masks, scaling, finite outputs and relevant gradients |
| FlashAttention | A specific fused implementation of attention | Actual dispatch, supported dtype/head sizes, numerical parity, backward where training needs it |
| xFormers-compatible paths | Existing native attention work, not an automatic import of the Python xFormers package | Exact operations and layouts executed by the selected backend |
| Flex Attention | Programmable score/mask behavior and its compiler/runtime dependencies | Preserve each requested modification; do not substitute ordinary dense attention silently |
| SageAttention | Quantized attention with architecture and numerical restrictions | Quantization/scaling contract, actual kernel used, model-quality assessment |
| Split attention | Partitioning and correct recombination of attention | Stable softmax rescaling across partitions, masks and head mapping |
| Sol-Attn | Approximate sparse inference attention described below | Eligible model integration, true sparse dispatch, dense-reference comparisons and output-quality checks |

An attention name is not itself a GPU backend. GGML CPU, Vulkan, CUDA, and other
providers supply execution; attention selection still depends on the operation,
shape, dtype, device, and whether gradients are required.

## NVIDIA Sol-Attn

The name is **Sol-Attn**, expanded by its authors as “Sparsifying online
attention.” The [paper](https://arxiv.org/abs/2607.24027), submitted July 27,
2026, targets image/video-generation inference. It chooses which key/value
blocks to evaluate in detail while reusing proxy information for other blocks.
This is an approximation with a quality/performance tradeoff; a dense fallback
must not be reported as successful Sol-Attn execution.

Source snapshot reviewed:
[NVlabs/Sana, sol-engine, 670482d8a857d578ac8a2ea89b052d0fb47badba](https://github.com/NVlabs/Sana/tree/670482d8a857d578ac8a2ea89b052d0fb47badba).

### Hardware: the current method is not Hopper-only

The official [kernel guide](https://github.com/NVlabs/Sana/blob/670482d8a857d578ac8a2ea89b052d0fb47badba/site_docs/techniques/sparse/sol_attn.md)
and [backend documentation](https://github.com/NVlabs/Sana/blob/670482d8a857d578ac8a2ea89b052d0fb47badba/techniques/sparse_backends/README.md)
list the following upstream implementations:

| Architecture | Example | Upstream execution path |
| --- | --- | --- |
| SM80 | NVIDIA A100 | Triton |
| SM89 | NVIDIA RTX 4090 | CuTe DSL |
| SM90 | NVIDIA H100 | CuTe DSL, with additional split-KV options |
| SM100 | NVIDIA B200/GB200 | CuTe DSL |
| SM120 | NVIDIA RTX 5090 | CuTe DSL |
| Apple Silicon | M-series | Metal through the specified PyTorch shader interface |

This table records upstream support, not support in our executable.
NVIDIA also publishes an [H100/A100 deployment comparison](https://nvlabs.github.io/Sana/Sol-Engine/H3-DataCenter/)
that explicitly distinguishes the SM90 CuTe path from the SM80 Triton path.
Do not copy the full-stack speedup numbers into an attention-only performance
claim: those runs also include other optimizations and particular model,
resolution, denoising and multi-GPU settings.

### Tensor and runtime contract

The reviewed upstream kernel interface is forward-only, uses contiguous BF16
Q/K/V in BTHD order, and specifies head dimension 128. Its Python interface
depends on PyTorch, with Triton/CUDA and optional CuTe DSL for NVIDIA execution,
or the documented PyTorch Metal shader interface for Apple. The pinned guide
lists Python 3.10+, PyTorch 2.10+, CUDA 12.8+, and Triton 3.6+ for the NVIDIA
path, with additional CuTe/CUTLASS Python and cuda-python requirements.

The upstream exact-sink option preserves selected KV blocks, while the model
adapter separately controls dense text-query behavior. These are distinct
requirements. Padding, token ranges, layout changes, threshold settings and
model-specific dense warmup must survive any native rewrite.

### Training implications

“Training-free” describes applying the inference optimization without retraining
the model. It does not mean the published kernel supplies the backward pass
needed by our SD 1.5 trainer.

Until a mathematically specified backward implementation is present and tested,
Sol-Attn is ineligible for training graphs. The native trainer must continue to
use an implementation with the required gradients. Do not detach tensors,
replace gradients with zeros, or silently route training through a forward-only
kernel.

### Work required for Cosmopolitan

There is no direct static C library drop-in in the reviewed upstream package.
A native integration must choose and implement a concrete route:

1. Define a C-facing operation contract over the shared GGML tensors, including
   layout, dtype, scaling, mask/sink semantics, threshold policy and ownership.
2. Implement a small CPU reference for correctness comparisons if a native
   sparse operation is pursued; do not label generic CPU attention as Sol-Attn.
3. Port or precompile the appropriate GPU kernels and provide a validated
   runtime launcher. Removing Python imports alone does not remove Triton,
   CuTe, PyTorch, compiler or device-runtime dependencies.
4. Treat Vulkan/SPIR-V as its own implementation effort. CUDA/CuTe kernels do
   not become Vulkan kernels by linking a Vulkan loader or wgpu-native.
5. Record actual provider selection and explicit fallback reasons. Validate
   approximate outputs and model quality against the dense path, and benchmark
   cold compilation separately from warm inference.
6. Test the same application bytes on Windows and Linux on supported hardware.
   A software Vulkan test cannot establish Hopper instructions or GPU speedups.

No Sol-Attn code has been imported in this milestone. The integration remains
an explicit work item, with inference and training support tracked separately.
