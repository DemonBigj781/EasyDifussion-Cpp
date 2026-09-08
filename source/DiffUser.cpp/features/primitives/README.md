# Primitive feature layout

Primitives are the project-wide prepared operations consumed by INFERENCE.cpp.
They are not limited to VAE work. DiffUser owns their normalized capability,
resource preparation, backend dispatch, and backend/device implementation;
INFERENCE owns the model logic that combines them.

## Canonical paths

```text
features/primitives/
├── common/devices/                 # final normalized contracts and registry
└── [backend]/
    └── [device]/
        ├── definition/[primitive_type].cpp
        └── translation/[primitive_type].cpp
```

Examples:

- `cpu/cpu/definition/strings.cpp`
- `cpu/cpu/translation/integer32.cpp`
- `cuda/gpu/definition/integer32.cpp`
- `cuda/gpu/translation/float16.cpp`
- `oneapi/gpu/translation/bfloat16.cpp`
- `openvino/npu/definition/integer8.cpp`

Each definition owns native behavior for one backend/device/type. Its matching
translation adapts and registers that behavior with `common/devices`. Source
backend and device differences are not visible to callers. A file may expose
several operations for one data type, but it must advertise only operations
that are actually implemented and tested. Unsupported combinations remain
absent; an empty file or directory never counts as support.

## Primitive types

The initial project-wide inventory is:

- `strings`
- `boolean`
- `bytes`
- `integer8`, `integer16`, `integer32`, `integer64`
- `unsigned8`, `unsigned16`, `unsigned32`, `unsigned64`
- `float16`, `bfloat16`, `float32`, `float64`
- quantized types as separate explicitly named contracts when their storage and
  arithmetic semantics are established

GPU backends are not required to implement `strings`. Text tokenization remains
host inference policy unless a real device implementation is intentionally
developed.

## Operation families

Every primitive type declares support independently for applicable operations:

- storage, allocation, copy, transfer, fill, and cast;
- shape/view, reshape, transpose, permute, slice, concat, repeat, pad, and roll;
- add, subtract, multiply, divide, scale/bias, clamp, and comparison;
- exp, log, square, square root, trigonometric operations, and reductions;
- activation functions;
- matrix multiplication and linear operations;
- convolution, pooling, interpolation, and upscaling;
- normalization;
- indexing, gather, top-k, and masking;
- positional/timestep embedding primitives;
- graph submission and synchronization bookkeeping.

Attention remains under `features/attention/`, not duplicated here. Composite
model blocks, samplers, VAE encode/decode, and completed-result delivery remain
INFERENCE responsibilities.

See the dated project-wide primitive audit under `Audit/2026-09-08/` before
adding a backend/type claim.
