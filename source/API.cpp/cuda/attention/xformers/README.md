# xFormers-compatible CUDA attention

This directory owns the ggml-native, inference-only memory-efficient attention forward path. It does not depend on PyTorch/ATen and does not provide a training/backward API.

The CUDA implementation accepts `GGML_OP_FLASH_ATTN_EXT` tensors with:

- F32, F16, or BF16 Q/K/V rows and F32 output;
- independent query/key and value head dimensions up to 512 values;
- broadcasted GQA/MQA K and V heads and batches;
- broadcasted F16 additive masks and ALiBi scaling;
- logit soft-capping and F32 attention sinks; and
- F32 online-softmax accumulation without materializing the score matrix.

Unsupported devices, dtypes, dimensions, precision modes, and layouts are rejected by `ggml_cuda_xformers_attn_supported()` so normal ggml dispatch can fall back safely. CUDA Pascal or newer is required. HIP and MUSA currently report unsupported.
