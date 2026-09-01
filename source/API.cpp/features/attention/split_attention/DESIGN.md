# Split Attention Translation and Unification Design

## Status

Design-only. This document defines the intended Split Attention architecture before backend definitions or common runtime code are written.

## Goal

Expose Split Attention to the application through one common feature API while allowing each compute backend and device class to use its own native slicing, chunking, memory-budgeting, tensor-layout, synchronization, or equivalent procedure.

The application should eventually call only the Split Attention `common/` layer. Backend/device-specific behavior must first be translated into normalized definitions. Common is implemented only after those definitions are understood and compared.

## Background

Split Attention is treated as a separate attention feature beside xFormers. Its purpose is to reduce peak memory pressure by dividing attention work into smaller pieces instead of requiring the complete attention workload to be resident at once.

Historical Stable Diffusion implementations include multiple split-attention strategies, including Doggettx-style split attention, InvokeAI split attention, and an older v1 split-attention mode. These variants are evidence that Split Attention should be modeled as a semantic feature with backend-specific procedures rather than as one fixed kernel implementation.

## Canonical layout

```text
API.cpp/features/attention/split_attention/
├── common/
│   ├── plan.cpp
│   ├── split.cpp
│   ├── qkt.cpp
│   ├── mask.cpp
│   ├── softmax.cpp
│   ├── av.cpp
│   ├── merge.cpp
│   └── forward.cpp
├── cuda/
│   └── definition/
│       ├── gpu/
│       │   └── [method].cpp
│       ├── cpu/
│       │   └── [method].cpp
│       └── npu/
│           └── [method].cpp
├── rocm/
│   └── definition/
│       ├── gpu/
│       │   └── [method].cpp
│       ├── cpu/
│       │   └── [method].cpp
│       └── npu/
│           └── [method].cpp
├── oneapi/
│   └── definition/
│       ├── gpu/
│       │   └── [method].cpp
│       ├── cpu/
│       │   └── [method].cpp
│       └── npu/
│           └── [method].cpp
└── cpu/
    └── definition/
        └── cpu/
            └── [method].cpp
```

Only device classes actually supported by a backend should eventually contain functional definitions. Empty structural branches may exist during the planning stage.

## Core rule

Translate first. Unify second.

A backend/device definition answers: "How does this target perform this normalized Split Attention operation, including any required chunking, slicing, temporary storage, synchronization, or workaround?"

A `common/` file answers: "Given normalized Split Attention definitions, what is the single application-facing behavior for this operation?"

Common must not assume CUDA, ROCm, oneAPI, or any historical implementation is the universal model.

## Proposed normalized methods

These are design candidates, not frozen ABI names.

### `plan.cpp`

Determines how the attention workload should be divided before execution.

Potential normalized inputs include tensor dimensions, head count, dtype, available memory/workspace, target backend/device, and requested limits.

Potential normalized outputs include split axis, number of slices/chunks, chunk sizes, workspace requirement, and whether a fused or unsplit path is preferable.

Backend definitions may use very different heuristics. The normalized result should describe the plan without exposing backend-native allocator or runtime objects.

### `split.cpp`

Creates or describes the logical slices/chunks consumed by the attention operation.

The split may occur across query positions, key/value positions, heads, batches, or another dimension when semantic equivalence is maintained. A backend may avoid physically copying data and instead expose views/offsets.

### `qkt.cpp`

Computes Query × Key-transpose scores for one logical split or chunk.

The definition must describe accumulation precision, scaling, layout expectations, and whether partial results are materialized or streamed.

### `mask.cpp`

Applies the normalized attention mask or bias for a split. Definitions must preserve global mask semantics even when only a subset of the score matrix is processed at a time.

### `softmax.cpp`

Normalizes attention scores while preserving correctness across splits.

This is a key design area: if the softmax domain is divided, a backend may need running maxima, running sums, rescaling, or another numerically stable merge procedure so the result matches unsplit attention semantics.

### `av.cpp`

Applies normalized attention weights to Value data for the current split. Definitions may accumulate partial output directly rather than materializing an intermediate attention matrix.

### `merge.cpp`

Combines partial Split Attention results into the final normalized result.

Depending on the split strategy, this may mean concatenation, accumulation, renormalization, rescaling, or no explicit operation if the backend streams directly into the final output.

### `forward.cpp`

Represents the complete Split Attention operation. A backend may implement the entire operation with a fused/native routine and document how that routine is semantically equivalent to `plan`, `split`, `qkt`, `mask`, `softmax`, `av`, and `merge`.

## Important distinction: split strategy versus public semantics

Split Attention should not expose every backend-specific chunking trick as a public API.

For example, one backend may split only queries, another may split key/value data, and another may use a streaming online-softmax algorithm. Those differences belong in backend definitions unless Common genuinely needs to control them.

The public/common behavior is simply: perform attention with reduced peak memory while preserving the normalized output semantics.

## Memory-budget behavior

Split Attention exists primarily to control peak memory use, so definitions should document:

- how available memory or workspace is estimated;
- minimum usable chunk size;
- alignment requirements;
- temporary-buffer lifetime;
- whether memory estimation is conservative or exact;
- whether a target can dynamically reduce chunk size after allocation failure;
- whether execution can proceed without a full score matrix;
- when the unsplit path is preferable.

Common may later expose a normalized memory budget or policy, but backend-native allocator details must remain inside definitions.

## Correctness requirements across splits

Definitions should aim to preserve the semantics of ordinary attention, including:

- Q/K/V dimensions and head mapping;
- scaling of QK^T;
- masks and additive biases;
- causal or structured mask behavior where applicable;
- GQA/MQA behavior where applicable;
- numerical stability of softmax;
- accumulation precision;
- output ordering;
- deterministic behavior where the backend can provide it.

A split implementation must not silently change the attention domain merely because work is processed in pieces.

## Potential C++ shape

The exact ABI is intentionally undecided. One possible normalized direction is:

```cpp
struct AttentionRequest;
struct SplitAttentionPlan;
struct SplitAttentionSlice;
struct AttentionResult;

namespace split_attention::definition::rocm::gpu {
    bool make_plan(const AttentionRequest &, SplitAttentionPlan &);
    bool execute_slice(const AttentionRequest &,
                       const SplitAttentionPlan &,
                       const SplitAttentionSlice &,
                       AttentionResult &);
}

namespace split_attention::common {
    bool forward(const AttentionRequest &, AttentionResult &);
}
```

This is illustrative only. The final interface must be derived after backend definitions have been studied.

## Variant handling

Historical Split Attention variants should initially be documented as implementation strategies, not separate application-facing APIs.

Potential strategies include:

- Doggettx-style split attention;
- InvokeAI-style split attention;
- older/v1 split attention;
- backend-native chunked attention equivalents;
- streaming/online-softmax equivalents where they preserve the same semantic goal.

If later analysis shows that two variants have materially different externally visible behavior or controls, they can become explicit normalized strategies. Do not create separate Common APIs solely because historical implementations used different internal algorithms.

## Capability states

As with xFormers, capability should not be represented by one boolean when hardware is unavailable for testing. Useful states may include:

- not compiled;
- compiled but unavailable;
- available but unvalidated;
- validated;
- unsupported for this request;
- supported only under a particular split strategy or memory limit.

## Development procedure

1. Create the folder and file skeleton.
2. Do not implement Split Attention Common yet.
3. Identify historical/reference Split Attention procedures.
4. Analyze one backend + device class at a time.
5. Translate each method into normalized backend definitions.
6. Document how each target chooses its split plan and preserves global softmax semantics.
7. Compare definitions across targets.
8. Identify genuinely shared behavior.
9. Design the Common contract from those translations.
10. Implement `common/[method].cpp` only after the definitions are sufficiently complete.
11. Route the application to Common only after Common is stable.
12. Validate memory usage, numerical correctness, and performance independently per backend/device class.

## Constraints for AI/code agents

- Do not copy the xFormers method implementation merely because some mathematical stages overlap.
- Do not assume one historical Split Attention variant defines the universal algorithm.
- Do not implement Common before backend/device definitions are analyzed.
- Do not modify existing attention implementations unless explicitly assigned.
- Backend-specific chunking, allocation recovery, synchronization, and workarounds belong in definitions.
- Common should operate on normalized concepts, not CUDA/HIP/SYCL-native objects.
- Compile success is not runtime validation.
- Preserve output semantics across splits; reduced memory usage is not permission to approximate silently.

## Open questions

- Which dimensions should be legal normalized split axes?
- Should Common choose the split axis, or should each definition choose it internally?
- Should memory budget be a hard byte limit, a policy, or both?
- How should online/running softmax state be represented without exposing backend-native types?
- When splitting K/V, what normalized merge state is required for exact softmax semantics?
- Are Doggettx, InvokeAI, and v1 sufficiently equivalent to remain internal strategies?
- Should allocation failure trigger automatic re-planning with smaller chunks?
- How should asynchronous execution and temporary-buffer lifetime be represented?
- Which methods need to be individually callable versus only documented as logical stages of `forward`?

The answers should come from translated backend/device definitions before Common is implemented.
