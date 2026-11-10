# xFormers feature

This directory owns the normalized xFormers-compatible attention contract and its backend implementations.

The required call direction is:

```text
caller -> common -> backend translation -> backend definition/native code
```

`common/xformers.hpp` defines `AttentionRequest`, capability reporting, validation, staged callbacks, fused-forward dispatch, and translation registration. Common contains no CUDA runtime types or launches.

The CPU translation registers staged `QKT -> mask -> softmax -> AV`
callbacks that adapt Common values to the CPU definition. The CUDA GPU
translation registers one fused-forward adapter; the fused QKT, mask, softmax,
and AV execution is owned by `cuda/definition/gpu/forward.cu` and is not
advertised as separately materialized stages.

Canonical ownership:

- `common/`: backend-neutral contract and dispatch;
- `[backend]/definition/[device]/`: backend capability and request rules;
- `[backend]/translation/[device]/`: normalized-to-native conversion,
  definition invocation, and native-to-normalized result conversion;
- `prototype/`: excluded reference and superseded code;
- `native-calls/`: research inventory, not an application API.

Post-relocation runtime coverage includes CPU F32 staged attention and CUDA
fused attention with F32/F16/BF16 Q/K/V, F32 output, byte strides, additive masks,
causal masking, explicit or maximum-bias ALiBi, logit soft-cap, attention
sinks, GQA/MQA, head and batch broadcasting, opaque CUDA streams, and
rejection paths. Both Common routes pass 32 model-byte lifecycle cycles; the
CUDA route also passes Compute Sanitizer memcheck with zero errors on an RTX
3060 (`sm_86`, driver 580.94.18, CUDA 12.4).

The historical stable-diffusion.cpp GGML adapter was a separate compatibility
adapter into this same Common contract, not a second backend implementation.
It was removed when commit `d883d6e` isolated DiffUser from SDKIT3, so the
current snapshot has no live application adapter and does not claim
end-to-end maturity.

Standalone tests live under `source/API.test/MultiTest/xformers/`; see `source/API.test/README.md` for build and execution commands.

Only the CPU/CUDA Common cycle sources are currently active. The image-generator
sources are retained as an integration harness but cannot link until an
explicit API.test-owned GGML adapter is restored. The Mesa/OpenGL/Vulkan files
and the shared cycle harness files are reserved placeholders and do not
establish xFormers support for those backends.

The retired materialized CUDA stages now live as `prototype/cuda-*.cu`. They are
not production translations and must not be added back to a build. See
[the dated layout audit](../../../../../Audit/2026-09-06/LAYOUT_AUDIT.md) for
the full source-status map.
