# xFormers

## Status
**Implemented — ggml-native CUDA forward path.**

The API.cpp implementation provides fused, memory-efficient inference attention without a PyTorch/ATen dependency. It supports F32, F16, and BF16 Q/K/V tensors; independent K/V dimensions; broadcasted GQA/MQA heads and batches; additive F16 masks with ALiBi; logit soft-capping; and attention sinks. Unsupported inputs report false so ggml can select its normal fallback.

## Validation
The source and build wiring are complete. Compilation and runtime comparison against the CPU reference remain required in the selected GitHub Actions CUDA environment.
