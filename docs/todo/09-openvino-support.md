# 09 — OpenVINO Support

## Objective
Use OpenVINO for graph-oriented inference on supported Intel CPU, GPU, and NPU devices where it provides a useful execution path.

## Implementation
1. Create an OpenVINO service/adapter rather than pretending it is identical to a GGML backend.
2. Start with bounded components that map cleanly to compiled graphs: YOLO, classifiers, CLIP/text encoder, or VAE.
3. Load OpenVINO IR directly and optionally support ONNX/model conversion through installed OpenVINO tooling.
4. Add device discovery (`CPU`, `GPU`, `NPU`, `AUTO`) and expose it in settings.
5. Cache compiled models by model hash, device, dtype, and shape policy.
6. Handle dynamic shapes deliberately; reject unsupported dynamic dimensions instead of recompiling unpredictably every request.
7. Define tensor-layout/dtype conversion boundaries between OpenVINO and the native diffusion pipeline.
8. Expand to larger diffusion subgraphs only after smaller components are validated and performance justifies the complexity.

## Dependencies
Can complement oneAPI rather than replace it. OpenVINO is especially useful for accelerator-specific graph execution and auxiliary models.

## Failure behavior
OpenVINO component failure should fall back to the native component only if tensor semantics are identical and the fallback is announced.

## Validation
Known-input output comparisons, compile cache reuse, device switching, dynamic/static shape cases, repeated load/unload, and integration with one pipeline component.

## Complete when
At least one meaningful Easy Diffusion component can be selected to run through OpenVINO with verified output and device placement.
