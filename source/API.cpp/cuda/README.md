# CUDA backend

NVIDIA CUDA implementations for the common API live here. Architecture-specific code may be split further by SM/device family when needed, while exposing one common API contract upward.

`attention/flash`, `attention/sage`, and `attention/xformers` are the canonical source locations compiled by the ggml CUDA build. The small headers directly under `cuda/` bridge shared ggml CUDA primitives that are not attention-owned.
