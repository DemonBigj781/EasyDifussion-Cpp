# 16 — Multi-GPU K-Sampling

## Objective
Use multiple GPUs during sampling/denoising while preserving correct scheduler and sampler behavior.

## Implementation strategy
1. Separate four different problems: independent-image parallelism, batch/latent sharding, pipeline/layer model parallelism, and tensor parallelism. Do not call all of them “multi-GPU sampling.”
2. Implement independent samples/seeds across GPUs first; it provides throughput gains with minimal synchronization.
3. For a single image, profile the denoiser graph and choose transfer boundaries with high compute-to-transfer ratio, such as block/layer groups.
4. Keep scheduler/K-sampler state authoritative on a coordinator and define exactly which tensors cross devices each step.
5. Maintain per-device model/activation ownership and separate VRAM budgets.
6. Support host-staged copies; peer-to-peer must be an optional optimization after capability testing.
7. Preserve deterministic single-device mode and add comparison tests for multi-device execution.
8. Extend backend abstraction so CUDA, SYCL, ROCm, and mixed-device experiments can use the same scheduling concepts without assuming identical memory APIs.

## Risks
Fine-grained splitting can be slower than one GPU because K-sampling repeats transfers every step. Optimize only after profiling.

## Validation
Correct seeds/scheduler progression, single-vs-multi output tolerance, multi-image throughput, one-image latency, OOM recovery, device loss/fallback, and transfer accounting.

## Complete when
At least one multi-GPU strategy gives a measurable real workload benefit with validated output and clear UI/device controls.
