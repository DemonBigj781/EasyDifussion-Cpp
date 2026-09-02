# CUDA backend

NVIDIA CUDA implementations for the common API live here. Architecture-specific code may be split further by SM/device family when needed, while exposing one common API contract upward.

The canonical implementations are the `cuda/translation/gpu` directories under the `flash`, `sage`, and `xformers` attention features. Compatibility headers retained here are Library-boundary notes only; production source ownership remains in feature translations.
