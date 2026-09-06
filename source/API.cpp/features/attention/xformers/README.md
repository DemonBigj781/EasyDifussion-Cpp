# xFormers feature

This directory owns the normalized xFormers-compatible attention contract and its backend implementations.

The required call direction is:

```text
caller -> common -> backend translation -> backend definition/native code
```

`common/xformers.hpp` defines `AttentionRequest`, capability reporting, validation, staged callbacks, fused-forward dispatch, and translation registration. Common contains no CUDA runtime types or launches.

The CPU translation registers a staged `QKT -> mask -> softmax -> AV` implementation. The CUDA GPU translation registers one fused-forward callback; its QKT, mask, softmax, and AV behavior is implemented inside that fused operation and is not advertised as separately materialized stages.

Canonical ownership:

- `common/`: backend-neutral contract and dispatch;
- `[backend]/definition/[device]/`: backend capability and request rules;
- `[backend]/translation/[device]/`: normalized-to-native conversion and execution;
- `prototype/`: excluded reference and superseded code;
- `native-calls/`: research inventory, not an application API.

Current runtime-tested coverage is CPU F32 staged attention and CUDA fused
attention with F32/F16/BF16 Q/K/V, F32 output, byte strides, additive masks,
causal masking, explicit or maximum-bias ALiBi, logit soft-cap, attention
sinks, GQA/MQA, head and batch broadcasting, opaque CUDA streams, and
rejection paths. Stable-diffusion.cpp converts GGML tensors into this same
Common contract; it is not a second backend implementation.

Standalone tests live under `source/API.test/MultiTest/xformers/`; see `source/API.test/README.md` for build and execution commands.

Only the CPU/CUDA cycle and generator sources are active. The Mesa/OpenGL/
Vulkan files and the shared cycle harness files are reserved placeholders and
do not establish xFormers support for those backends.

The retired materialized CUDA stages now live as `prototype/cuda-*.cu`. They are
not production translations and must not be added back to a build. See
`../../../LAYOUT_AUDIT.md` for the full source-status map.
