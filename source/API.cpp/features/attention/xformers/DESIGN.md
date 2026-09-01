# xFormers Feature Translation and Unification Design

## Status

Design-only. This document defines the intended architecture and implementation procedure before backend definitions or common runtime code are written.

## Goal

Expose xFormers-compatible attention to the application through one common feature API while allowing each compute backend to use its own native calls, kernels, workarounds, data layouts, or equivalent procedures.

The application should call only the common layer. Backend-specific details must be translated first into normalized definitions. The common layer is built only after those backend definitions are understood.

## Canonical layout

```text
API.cpp/features/attention/xformers/
├── common/
│   ├── qkt.cpp
│   ├── mask.cpp
│   ├── softmax.cpp
│   ├── av.cpp
│   └── forward.cpp
├── cuda/
│   └── definition/
│       ├── qkt.cpp
│       ├── mask.cpp
│       ├── softmax.cpp
│       ├── av.cpp
│       └── forward.cpp
├── rocm/
│   └── definition/
│       ├── qkt.cpp
│       ├── mask.cpp
│       ├── softmax.cpp
│       ├── av.cpp
│       └── forward.cpp
└── oneapi/
    └── definition/
        ├── qkt.cpp
        ├── mask.cpp
        ├── softmax.cpp
        ├── av.cpp
        └── forward.cpp
```

Additional backends may mirror the same pattern.

## Core rule

Translate first. Unify second.

A backend `definition/` file answers: "What does this normalized xFormers operation mean on this backend, and how can this backend produce equivalent behavior?"

A `common/` file answers: "Given normalized backend definitions, what is the single application-facing behavior for this xFormers operation?"

The common layer must not be designed by assuming CUDA is the universal model. CUDA, ROCm, oneAPI, and future backends are peers whose definitions are compared before common behavior is finalized.

## Definition layer

A backend definition may contain more than a declaration. It may include backend-specific translation code, adapters, helper calls, alternate procedures, emulation, or roundabout logic when the backend does not expose a direct equivalent operation.

The definition layer should normalize:

- operation meaning;
- input and output expectations;
- dtype and shape requirements;
- mask or bias behavior;
- backend-native call sequence;
- memory/layout assumptions;
- synchronization requirements;
- unsupported cases;
- fallback or equivalent procedures;
- capability and validation status.

The goal is semantic equivalence, not identical implementation.

## Common layer

The application calls only `common/` methods. Common owns the full application-facing operation and may select the appropriate normalized backend definition at runtime or compile time.

Common should contain no backend-native kernel names, HIP launch details, CUDA-specific calls, or SYCL-specific objects unless a future design explicitly proves they are genuinely universal abstractions.

Common may perform:

- backend selection;
- capability checks;
- normalized argument validation;
- common preprocessing/postprocessing;
- operation sequencing;
- fallback selection;
- error normalization;
- common result handling.

## Proposed method meanings

These names are structural starting points, not frozen ABI names.

### `qkt.cpp`

Normalized Query × Key-transpose score generation. Backend definitions may tile, split, fuse, or otherwise avoid materializing the complete score matrix as long as the externally visible semantics match.

### `mask.cpp`

Normalized attention mask/bias application. Backend definitions describe supported mask forms and how native backend mechanisms reproduce the required behavior.

### `softmax.cpp`

Normalized attention-score normalization. Definitions describe backend-specific precision, accumulation, tiling, and stability behavior required to produce equivalent results.

### `av.cpp`

Normalized attention-weight × Value application. Definitions may fuse this with surrounding work internally if the normalized result remains equivalent.

### `forward.cpp`

Normalized complete forward attention operation. A backend may expose a fused native forward path instead of separately executing QKT, mask, softmax, and AV. The definition should state that equivalence explicitly.

## Important design principle: logical methods are not mandatory kernel boundaries

The file structure describes normalized concepts, not necessarily separate GPU launches. A backend may implement `forward` with one fused kernel while still documenting how its behavior corresponds to QKT, mask, softmax, and AV.

Likewise, if a backend requires Split-K, tiling, streaming softmax, temporary buffers, or another algorithmic technique, that may remain internal to the relevant definition unless Common truly needs to select or control it as a public capability.

## Potential C++ shape

The exact ABI is intentionally undecided. A possible direction is:

```cpp
struct AttentionRequest;
struct AttentionResult;
struct XFormersCapabilities;

enum class XFormersBackend {
    CUDA,
    ROCM,
    ONEAPI,
    CPU,
};

namespace xformers::definition::cuda {
    bool supports_qkt(const AttentionRequest &);
    bool qkt(const AttentionRequest &, AttentionResult &);
}

namespace xformers::common {
    bool qkt(const AttentionRequest &, AttentionResult &);
}
```

This is illustrative only. The final interface should be derived after backend definitions have been analyzed.

## Capability states

Do not reduce capability to a single boolean when the project cannot test every backend immediately. A useful model may distinguish:

- not compiled;
- compiled but unavailable;
- available but unvalidated;
- validated;
- unsupported for this request.

This prevents a compile-successful ROCm path from being misrepresented as runtime-proven.

## Development procedure

1. Create the folder and file skeleton.
2. Do not implement Common yet.
3. Analyze one backend method at a time.
4. Fill each backend's `definition/[method].cpp` with the normalized meaning and backend-specific translation/equivalent procedure.
5. Compare definitions across backends.
6. Identify behavior that is actually shared.
7. Design the Common contract from those normalized definitions.
8. Implement `common/[method].cpp` only after the definitions are sufficiently complete.
9. Route the application to Common only after Common is stable.
10. Validate runtime behavior independently per backend.

## Constraints for AI/code agents

- Do not modify existing CUDA/ROCm/oneAPI attention implementations merely to satisfy this new tree unless explicitly assigned.
- Do not copy CUDA assumptions into Common without comparing other backend definitions.
- Do not implement Common while backend definitions are still unknown.
- Do not claim runtime validation from compile success alone.
- Backend-specific workarounds belong in that backend's definition layer.
- Application code should eventually depend on Common, not backend-native calls.
- Preserve existing working or in-progress API migration code unless a task explicitly targets it.

## Open questions to resolve during definition work

- Which xFormers operations must be first-class normalized methods versus internal implementation details?
- Is a complete fused `forward` definition sufficient for some backends, or must component-level methods also be callable?
- Which tensor/layout representation should Common use?
- How should masks, biases, GQA/MQA, ALiBi, soft-capping, and attention sinks be normalized?
- What precision differences are acceptable across backends?
- How should unsupported shapes or dtypes report fallback?
- Which capabilities are compile-time versus runtime detectable?
- Which backend owns allocation/workspace lifetime?
- How should asynchronous execution and synchronization be represented without leaking backend-native objects into Common?

The answers should come from backend definitions first, not from assumptions made in Common.
