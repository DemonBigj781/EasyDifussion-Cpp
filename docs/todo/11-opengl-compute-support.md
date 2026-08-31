# 11 — OpenGL Compute Support

## Objective
Investigate OpenGL compute shaders as a last-resort compatibility backend on systems with modern OpenGL but without a practical compute API.

## Implementation
1. Require a version/profile supporting compute shaders and shader storage buffer objects.
2. Build a proof-of-concept backend with context ownership, SSBO allocation, tensor upload/download, compute dispatch, barriers, and error reporting.
3. Implement representative kernels first: elementwise op, normalization, and a tiled matrix multiply.
4. Measure shader compile overhead, memory limits, synchronization cost, and driver portability before attempting a full model.
5. Add shader cache keyed by GPU/driver/source options.
6. If the proof-of-concept is viable, create a constrained tensor-operation interface compatible with the model backend abstraction.
7. Do not make OpenGL a dependency of headless/server builds.

## Risks
OpenGL is graphics-first, driver behavior varies widely, and ML dtypes/subgroup operations may be poor. The investigation can legitimately conclude that full diffusion inference is not worthwhile.

## Validation
Kernel correctness, large-buffer behavior, headless context if needed, repeated context teardown, and performance comparison against CPU/OpenCL.

## Complete when
A documented proof-of-concept determines viability. Full-model support is a second milestone only if the prototype is technically justified.
