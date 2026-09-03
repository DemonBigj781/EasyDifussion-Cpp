# xFormers execution model and unified interpreter plan

## Purpose

This document records how xFormers-compatible attention is executed by each backend/device architecture in the `Theory` branch and defines the normalization layer needed to make those implementations look like one operation to the rest of API.cpp.

This is an execution-model document, not a claim that every backend is complete or runtime-validated.

The central rule is:

> **Common interprets one normalized xFormers request. Translations describe how a backend can satisfy that request. Definitions and native kernels perform the backend-specific work.**

The unified layer must normalize semantics without forcing every architecture to use the same kernel decomposition.

---

## Logical xFormers operation

At the semantic level, attention can be described as:

```text
Q, K, V
  |
  v
QK^T score generation
  |
  v
scale / bias / mask / causal handling
  |
  v
stable normalization (softmax)
  |
  v
weights * V
  |
  v
output
```

The current file vocabulary maps this into five logical methods:

- `qkt` — generate Query × Key-transpose scores;
- `mask` — apply additive mask, causal restriction, bias, or equivalent score modification;
- `softmax` — normalize scores stably;
- `av` — apply normalized attention weights to Value;
- `forward` — provide the complete operation and choose whether the backend uses staged, fused, tiled, streamed, or native-library execution.

These are **semantic stages**, not mandatory kernel boundaries.

---

## Current architecture execution survey

### Native/general CPU

Current implementation status: **working translation-level reference path; runtime validation still separate**.

The CPU translation currently implements the logical stages directly and materializes the complete score matrix:

```text
forward
  -> allocate q_tokens * kv_tokens score buffer
  -> qkt(...)
  -> apply_mask(...)
  -> softmax(...)
  -> av(...)
  -> output
```

Characteristics:

- input/output representation is currently plain `float` pointers;
- score storage is explicit and temporary;
- `forward` is orchestration rather than a fused kernel;
- causal behavior and additive masks are normalized at the mask stage;
- the CPU path is therefore useful as a semantic/reference decomposition, but it must **not** be treated as the required implementation strategy for accelerators.

The CPU path demonstrates the operation's meaning, not the only legal execution plan.

### CUDA / NVIDIA GPU

Current implementation status: **substantial native translation exists; runtime validation remains required**.

CUDA does not need to execute the five logical stages as five independent kernels. The current `xformers-attention.cu` implementation performs a fused memory-efficient attention loop.

Conceptually, each CUDA work row performs:

```text
select batch/head/query row
  |
  +-- for every K/V position --------------------+
  |                                               |
  |   compute Q dot K                             |
  |   apply scale / softcap                       |
  |   apply mask / ALiBi-style bias if present   |
  |   update online softmax state                 |
  |   update running weighted V accumulator       |
  |                                               |
  +-----------------------------------------------+
  |
  +-- optionally fold attention sink into denominator
  |
  v
normalize accumulated V by final softmax denominator
  |
  v
write output
```

Important CUDA details currently represented by the implementation:

- NVIDIA Pascal-or-newer capability check;
- F32, F16, and BF16 Q/K/V support in the current path;
- F32 output;
- grouped/multi-query head mapping through divisible head counts;
- batch broadcasting through divisible batch counts;
- optional F16 mask;
- optional F32 attention sinks;
- scale, maximum bias, and logit softcap parameters;
- online softmax, avoiding a full materialized score matrix;
- one fused forward path may satisfy `qkt`, `mask`, `softmax`, and `av` semantics together.

Therefore CUDA proves why Common cannot require a backend to expose separate physical kernels for each logical method.

### ROCm / AMD GPU and ROCm-family device classes

Current implementation status: **definition skeleton exists; translation is not implemented yet**.

The tree currently reserves CPU, GPU, and NPU device classes under the ROCm backend family. The five normalized method names exist as definition placeholders for CPU/GPU coverage, while translation directories are still structural placeholders.

The ROCm execution model must be documented from the actual HIP/ROCm implementation chosen later. It may use:

- a fused HIP kernel;
- rocBLAS/hipBLAS primitives plus normalization kernels;
- another ROCm-native attention primitive where available;
- staged emulation;
- architecture-specific alternatives selected by capability.

No one of those choices should be assumed until implemented and measured.

The interpreter contract should make any of them legal as long as their externally visible semantics match the normalized request.

### oneAPI / SYCL CPU, GPU, and possible NPU

Current implementation status: **definition skeleton exists; translations are not implemented yet**.

The tree already distinguishes oneAPI backend family from oneAPI device class. CPU and GPU definition placeholders exist and NPU is reserved separately.

This is important because `oneAPI` does not imply one execution architecture. A oneAPI CPU target and a oneAPI GPU target may choose completely different procedures while sharing the same backend family and normalized xFormers request.

Potential execution strategies can include:

- SYCL kernels;
- oneMKL-assisted matrix operations plus device kernels;
- vendor extension paths;
- fused kernels when supported;
- staged fallback when fusion is unavailable.

These are possibilities, not current validation claims.

### NPU targets

Current implementation status: **reserved only where represented in backend trees**.

An NPU must not be forced to imitate GPU kernel structure. If an NPU exposes one graph/operator that semantically performs attention, its translation should advertise a fused execution plan to Common. If only smaller primitives are available, its translation can advertise staged execution.

---

## Problem with backend-owned orchestration

If every backend owns its own public `forward` meaning, API.cpp eventually develops several subtly different xFormers implementations:

```text
application -> cuda forward
application -> rocm forward
application -> oneapi forward
application -> cpu forward
```

That creates duplicated policy for:

- input validation;
- shape normalization;
- dtype normalization;
- mask interpretation;
- causal interpretation;
- scale defaults;
- capability checking;
- fallback choice;
- error reporting;
- output contract;
- runtime test expectations.

Backend-specific execution should differ. Public semantics and selection policy should not.

---

## Proposed unified interpreter

“Interpreter” here means a **small normalized execution-plan interpreter**, not a heavyweight virtual machine and not a new tensor language.

The application submits one normalized request:

```text
XformersRequest
  Q
  K
  V
  optional mask
  optional sinks
  shape/head/batch metadata
  dtype metadata
  scale
  causal mode
  bias parameters
  logit-softcap parameters
  requested precision
```

Common validates and normalizes that request once.

The selected backend translation then exposes an execution description such as:

```text
XformersExecutionPlan
  backend
  device_class
  capabilities
  execution_kind
  native_context
  operations/callbacks
```

`execution_kind` should initially support at least:

```text
fused
staged
native_operator
unsupported
```

Meaning:

- **fused** — one backend operation satisfies the whole normalized forward contract, as the current CUDA kernel largely does;
- **staged** — Common interprets the normalized sequence and calls backend translations for `qkt`, `mask`, `softmax`, and `av`, similar to the current CPU decomposition;
- **native_operator** — a backend/runtime provides a native attention operator whose internal decomposition is opaque;
- **unsupported** — the translation cannot satisfy this request and Common may choose a documented fallback.

A future `hybrid` kind can be added only if a real backend requires a mixture such as fused QK/softmax plus separate AV. Do not add execution kinds speculatively.

---

## Intended call boundary

The desired direction is:

```text
Application / Library
        |
        v
xformers/common
  - normalize request
  - select backend/device
  - query capability/plan
  - interpret plan
  - normalize errors/results
        |
        v
backend translation
  - convert normalized request to backend representation
  - advertise supported execution kind
  - expose backend execution callbacks
        |
        v
backend definition/native implementation
  - CUDA/HIP/SYCL/CPU/NPU details
  - kernels, SDK calls, layouts, streams, queues
```

The reverse result path is:

```text
native result/status
  -> backend translation
  -> normalized Common result
  -> Library/Application
```

Backend-native objects must not leak above translation.

---

## What Common owns

Common should eventually own only architecture-neutral policy:

- normalized request structure;
- tensor metadata validation;
- normalized dtype/shape/head rules;
- semantic mask and causal rules;
- scale/bias/softcap interpretation;
- backend/device selection;
- capability query;
- execution-plan interpretation;
- fallback policy;
- normalized result/error state;
- cross-backend conformance expectations.

Common must **not** contain:

- CUDA launches;
- HIP launches;
- SYCL queues;
- vendor SDK handles;
- architecture-specific tile sizes;
- warp/wave/subgroup assumptions;
- backend-specific memory ownership.

---

## What a translation owns

Each translation answers two questions:

1. **Can this backend/device satisfy this normalized request?**
2. **If yes, which execution plan should Common interpret?**

A translation may therefore perform:

- capability checks;
- dtype/layout compatibility checks specific to the backend;
- conversion from normalized metadata to native parameter structures;
- native context/stream/queue adaptation;
- selection among backend-native kernels;
- callback registration for staged execution;
- fused/native operator dispatch.

The translation must not redefine what xFormers attention means.

---

## First normalized operation set

Do not begin with a large opcode language. The first interpreter contract should remain intentionally small:

```text
FORWARD_FUSED
QKT
APPLY_MASK
SOFTMAX
AV
```

The plan chooses either `FORWARD_FUSED` or the staged sequence.

This makes the current implementations representable immediately:

```text
CPU  -> QKT -> APPLY_MASK -> SOFTMAX -> AV
CUDA -> FORWARD_FUSED
ROCm -> unknown until translation is implemented
oneAPI CPU/GPU/NPU -> unknown until translation is implemented
```

If a later backend proves that this vocabulary is insufficient, extend it from observed requirements rather than designing speculative operations.

---

## Capability description

Execution-plan selection should be data-driven. A backend translation should be able to report constraints such as:

- supported Q/K/V/output dtypes;
- maximum head/value dimensions;
- mask support and mask dtype/layout;
- causal support;
- grouped-query/multi-query support;
- batch/head broadcast rules;
- sinks support;
- ALiBi/bias support;
- logit-softcap support;
- precision modes;
- alignment/stride requirements;
- architecture minimums;
- whether score materialization is required;
- whether execution is fused/staged/native;
- validation state: untested, compile-only, runtime-smoke, conformance-tested.

Common should consume these capabilities instead of hard-coding backend names into semantic behavior.

---

## Conformance rule

Backends do **not** have to produce bit-identical intermediate values or use identical arithmetic order.

They do have to satisfy the same externally visible contract within documented numeric tolerances:

```text
same normalized request
        |
        +--> CPU reference/staged result
        +--> CUDA fused result
        +--> ROCm result
        +--> oneAPI result
        |
        v
compare output shape, validity, semantics, and numeric tolerance
```

This is what Vast.ai runtime testing should eventually validate after the backend code is compiled on GitHub.

Compile success alone is not conformance.

---

## Development order

1. Record the actual execution behavior of every existing implementation.
2. Keep CPU staged execution and CUDA fused execution as the first two concrete models.
3. Define the smallest normalized request and execution-plan structures capable of describing both.
4. Implement ROCm and oneAPI translations by describing their real execution behavior rather than copying CUDA.
5. Add Common plan interpretation only after at least two distinct backend strategies can be represented.
6. Route Library/application calls to Common.
7. Compile the relevant feature code in GitHub CI.
8. After explicit approval, run the GitHub-built container/artifact on Vast.ai hardware.
9. Compare runtime behavior against the normalized contract.

---

## Current status matrix

| Backend family | Device class | Definition | Translation | Observed execution strategy | Runtime validation |
| --- | --- | --- | --- | --- | --- |
| CPU | CPU | incomplete/skeletal | implemented staged path | materialized `QKT -> mask -> softmax -> AV` | not established here |
| CUDA | GPU | incomplete/skeletal | substantial implementation | fused streaming attention with online softmax | still required |
| ROCm | GPU | skeletal | placeholder | not yet determined | not tested |
| ROCm | CPU | skeletal | placeholder | not yet determined | not tested |
| ROCm | NPU | reserved | placeholder | not yet determined | not tested |
| oneAPI | CPU | skeletal | placeholder | not yet determined | not tested |
| oneAPI | GPU | skeletal | placeholder | not yet determined | not tested |
| oneAPI | NPU | reserved | placeholder | not yet determined | not tested |

This table must be updated when implementation or runtime evidence changes.

---

## Architecture principle

The unified xFormers implementation should therefore be thought of as:

> **one semantic program, multiple execution plans.**

CPU may interpret that program as separate stages. CUDA may satisfy it with one fused kernel. ROCm or oneAPI may use yet another decomposition. Common owns the meaning and orchestration policy; translations describe the available plan; native backend code owns the mechanics.
