# xFormers native-call inventories

This directory records backend-native execution primitives **before** any attempt is made to choose a shared translation vocabulary.

## Required order of work

1. Inventory the calls, intrinsics, execution objects, memory spaces, launch forms, dtype conversions, synchronization, and error/reporting mechanisms actually used or required by each backend.
2. Record backend-specific semantic constraints and architecture restrictions.
3. Compare inventories side by side.
4. Only then choose normalized calls for each backend `translation/` directory.
5. Common consumes normalized translation calls only; Common must not directly depend on CUDA, HIP, SYCL, OpenCL, or other vendor syntax.

## Rule

These files are descriptive inventories, not translation specifications. Similar-looking calls must not be declared equivalent here merely because their names resemble one another.

Do not invent a normalized function name in these files. Do not use CUDA as the canonical source vocabulary. Absence of a feature on a backend is valid inventory data and should be documented rather than hidden by an assumed fallback.

## Inventories

- `cuda.md` — NVIDIA CUDA primitives currently used by the xFormers CUDA path plus nearby CUDA primitives that may become relevant to optimized implementations.
- `rocm-hip.md` — AMD ROCm/HIP execution and hardware primitives relevant to a future native implementation; placeholders are identified as such.
- `oneapi-sycl.md` — oneAPI/SYCL execution, group, memory, queue, and datatype primitives relevant to a future native implementation.
- `cpu.md` — exact host C++ operations used by the current staged CPU implementation plus clearly separated future optimization categories.

## Coverage checklist for each backend

Before an inventory is considered ready for cross-backend comparison, it should answer every applicable category below.

### Execution model

- host/device boundary;
- kernel/function entry mechanism;
- compilation/target annotations;
- execution object (stream, queue, ordinary call, etc.);
- asynchronous versus synchronous behavior.

### Work decomposition

- thread/work-item identity;
- block/work-group identity;
- grid/global-range identity;
- warp/wave/sub-group identity;
- assumptions about execution width.

### Memory spaces and addressing

- global/device/host memory;
- block/work-group local memory;
- private/register storage;
- allocation/lifetime mechanism;
- tensor pointer/stride/layout assumptions;
- staging or repacking requirements.

### Synchronization and ordering

- work-group/block barriers;
- sub-group/warp barriers;
- cross-lane movement;
- memory fences;
- host-side event/dependency primitives;
- stream/queue synchronization semantics.

### Arithmetic and reductions

- Q·K accumulation strategy;
- accumulator precision;
- reduction topology;
- matrix/tensor acceleration if used;
- scalar versus vectorized execution.

### Mask/bias behavior

- additive masks;
- causal masks;
- mask datatype/layout;
- broadcast/remapping rules;
- ALiBi or related bias;
- logit soft-cap;
- attention sinks.

### Softmax behavior

- materialized versus online;
- maximum reduction;
- exponential operation;
- denominator accumulation;
- special handling for NaN and infinities;
- fully masked rows;
- normalization precision;
- whether AV is fused with softmax.

### AV/output behavior

- output accumulation strategy;
- output accumulator precision;
- output datatype;
- write layout;
- ownership/race assumptions.

### Datatypes

- F32;
- F16;
- BF16;
- quantized forms if any;
- conversion intrinsics/routines;
- vector/packed forms if used.

### Atomics

- whether used;
- exact operations;
- datatype restrictions;
- memory-order/scope behavior where applicable.

### Math/special values

- exponential;
- max/min;
- sqrt;
- tanh/soft-cap operations;
- finite/NaN/infinity classification;
- reciprocal/division;
- any architecture-specific approximations.

### Capability selection

- architecture/device gates;
- layout gates;
- datatype gates;
- work-group/shared-memory limits;
- matrix-instruction capability;
- dispatch selection mechanism.

### Error/status handling

- pre-dispatch support checks;
- runtime error APIs;
- asynchronous error behavior;
- assertions/aborts/exceptions;
- diagnostic/status conversion.

### Integration boundary

- backend context object;
- execution stream/queue ownership;
- tensor representation received from the surrounding library;
- exact native functions/symbols called by the implementation.

### Validation state

Each inventory should distinguish among:

- present in current repository implementation;
- supported native primitive but not currently used;
- candidate for a future implementation;
- architecture-dependent;
- unverified/unknown;
- unsupported.

## Comparison readiness

Cross-backend comparison should not start merely because each file exists. It begins when the categories above are documented sufficiently to answer both of these questions for each backend:

1. **What operations does this backend implementation actually need to perform xFormers-compatible attention?**
2. **Which details are semantic requirements and which are only consequences of that backend's chosen implementation strategy?**

Only after that distinction is visible should a comparison table classify relationships as exact, parameterized, approximate, emulated, fused/split, backend-private, unsupported, or unknown.

The later comparison document should cite these inventories and explicitly state where behavior is exact, approximate, emulated, unsupported, architecture-dependent, or still unknown.
