# xFormers

## Status
**Source implemented — toolkit/runtime validation not recorded for this snapshot.**

The production route converts GGML tensors in the stable-diffusion.cpp application adapter, enters the backend-neutral Common contract, and dispatches to the registered fused CUDA translation and definition. It supports strided F32, F16, and BF16 Q/K/V, F32 output, broadcasted heads and batches, supported additive-mask dtypes, ALiBi, logit soft-cap, and attention sinks. Unsupported requests return false so the application can select its fallback.

## Validation
The source and build wiring are complete. Compilation and runtime comparison against the CPU reference remain required in the selected GitHub Actions CUDA environment.
