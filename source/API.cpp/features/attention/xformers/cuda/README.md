# CUDA xFormers methods

Structural placeholder for feature-first xFormers CUDA method implementations.

Canonical GPU method path: `API.cpp/features/attention/xformers/cuda/definition/gpu/[method].cpp`.
CUDA-to-common adapters and production kernels live under `cuda/translation/gpu/`.

Do not migrate, replace, or modify the existing CUDA attention/API implementation as part of creating this structure. Method files should be added only when their common contract is defined.
