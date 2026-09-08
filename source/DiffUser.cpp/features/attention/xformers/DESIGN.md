# xFormers Feature Translation and Unification Design

## Status

Implemented for the normalized CPU and CUDA Common routes. ROCm, oneAPI, and other backend translations remain design/structure work.

## Goal

Expose xFormers-compatible attention to the application through one common feature API while allowing each backend family and device class to use its own native calls, kernels, workarounds, layouts, or equivalent procedures.

Feature callers use only `common/`. Backend-native behavior is represented by
an exact backend definition and adapted through its translation. Standalone
CPU/CUDA tests prove that boundary, and the stable-diffusion.cpp GGML
application adapter has now exercised the same Common CUDA route during image
generation.

## Canonical layout

```text
DiffUser.cpp/features/attention/xformers/
├── common/
│   ├── qkt.cpp
│   ├── mask.cpp
│   ├── softmax.cpp
│   ├── av.cpp
│   └── forward.cpp
├── cuda/
│   └── definition/
│       └── gpu/
│           ├── qkt.cpp
│           ├── mask.cpp
│           ├── softmax.cpp
│           ├── av.cpp
│           └── forward.cpp
├── rocm/
│   └── definition/
│       ├── gpu/
│       │   └── [method].cpp
│       ├── cpu/
│       │   └── [method].cpp
│       └── npu/
│           └── [method].cpp when applicable
├── oneapi/
│   └── definition/
│       ├── gpu/
│       │   └── [method].cpp
│       ├── cpu/
│       │   └── [method].cpp
│       └── npu/
│           └── [method].cpp when applicable
└── cpu/
    └── definition/
        └── cpu/
            └── [method].cpp
```

Not every backend family must implement every device class. The device-class layer exists so one backend family may expose multiple independently translated execution targets without creating fake top-level backends.

The diagram above records definition coverage. Every backend also owns the parallel
`[backend]/translation/[device-class]/[method].cpp` path defined by `DiffUser.cpp/LAYOUT.md`.

## Core rule

**Translate first. Unify second.**

The backend-definition path is:

`[backend]/definition/[device-class]/[method].cpp`

A definition answers: "What does this normalized xFormers operation mean for this exact backend family and device class, and how can it produce equivalent behavior?"

The translation path is:

`[backend]/translation/[device-class]/[method].cpp`

A translation adapts that backend definition to the xFormers common contract.

A `common/[method].cpp` file answers: "Given the normalized definitions, what is the single application-facing behavior?"

## Backend family versus device class

Backend family and hardware target are different axes.

Examples:

- oneAPI may have separate GPU, CPU, and potentially NPU definitions;
- ROCm-family work may expose more than one device-class translation;
- CUDA currently maps to a GPU definition;
- the project's native/general CPU backend maps to a CPU definition.

Common should not care whether a normalized operation originated from a GPU, CPU, or NPU except where capability or selection policy requires that information.

## Definition layer

A definition may contain declarations, native API semantics, alternate procedures, emulation requirements, capability state, or roundabout behavior when the target does not expose a direct equivalent. Backend-to-common adapter code belongs in the parallel translation path.

Definitions should normalize operation meaning, inputs/outputs, dtype and shape rules, masking/bias semantics, memory/layout assumptions, synchronization, unsupported cases, fallbacks/equivalents, capability state, and validation state.

The goal is semantic equivalence, not identical native implementation.

## Translation layer

Translations implement the backend-to-common boundary. They may call native kernels,
SDK functions, or compatibility helpers, but must return normalized behavior to
`common/` without leaking backend-native objects into the Library API.

## Common layer

The application eventually calls only `common/`. Common owns backend selection, capability checks, normalized validation, shared sequencing, common preprocessing/postprocessing, fallback selection, error normalization, and result handling.

Common should not contain backend-native CUDA, HIP, SYCL, or other
device-specific calls. Native semantics belong in definitions and execution/
SDK adaptation belongs in translations.

## Current structural method vocabulary

These names are the current normalized semantic concepts; the contract is implemented but is not yet a frozen external ABI:

- `qkt.cpp` — Query × Key-transpose score generation;
- `mask.cpp` — normalized mask/bias behavior;
- `softmax.cpp` — normalized score normalization;
- `av.cpp` — attention-weight × Value application;
- `forward.cpp` — normalized complete forward attention behavior.

Logical methods do not imply separate native kernel launches. A backend/device definition may fuse, tile, split, stream, or emulate internally as long as it documents equivalent semantics.

## Development procedure

For a new backend/device target:

1. Analyze its real method and execution constraints independently.
2. Fill `[backend]/definition/[device-class]/[method].cpp` with native semantics and constraints.
3. Implement `[backend]/translation/[device-class]/[method].cpp` against the existing normalized contract.
4. Advertise only capabilities the translation actually implements.
5. Compare output behavior against the CPU staged reference within documented tolerances.
6. Extend Common only when an observed backend requirement cannot be represented safely.
7. Route application callers and any Library facade through Common after the translation is stable.
8. Compile and runtime-test independently on the target backend/device.

## Constraints for AI/code agents

- Do not modify existing in-progress attention implementations merely to satisfy this new tree unless explicitly assigned.
- Do not treat CUDA or any other backend as the universal source model.
- Do not flatten CPU/GPU/NPU targets when one backend family exposes multiple distinct definitions.
- Do not extend Common speculatively while a new backend/device definition is unknown.
- Do not claim runtime validation from compile success alone.
- Backend/device-specific workarounds belong in that exact definition path.
- Preserve existing working or interrupted migration code unless a task explicitly targets it.
