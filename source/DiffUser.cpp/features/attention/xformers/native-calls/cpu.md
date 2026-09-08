# CPU native-operation inventory for xFormers

Status: raw inventory. Do not choose normalized translation names from this file yet.

Purpose: record the host-language execution, memory, math, sequencing, datatype, layout, and error behavior actually used by the current CPU xFormers path, plus CPU-native optimization categories that must be inventoried separately if introduced later.

## Execution model

- Ordinary C++ function calls.
- Sequential orchestration in `forward(...)`.
- Raw pointer inputs/outputs for Q, K, V, mask, scores, and output.
- `std::vector<float>` for temporary materialized score storage.
- Scalar loops over query tokens, key/value tokens, and head dimension.
- No explicit task runtime, thread pool, OpenMP region, SIMD intrinsic layer, or asynchronous queue in the currently inspected implementation.

## Current logical execution sequence

The current CPU `forward(...)` performs:

1. validate required pointers and dimensions;
2. allocate `q_tokens * kv_tokens` F32 score storage;
3. call `qkt(...)`;
4. call `apply_mask(...)`;
5. call `softmax(...)`;
6. call `av(...)`;
7. return success.

The complete score matrix is materialized before mask, normalization, and AV.

## Current function-level operations

- `qkt(const float*, const float*, float*, ...)`
- `apply_mask(float*, const float*, ..., bool causal)`
- `softmax(float*, ...)`
- `av(const float*, const float*, float*, ...)`
- `forward(...)`

These are current CPU translation functions, not yet proposed normalized calls.

## QKT implementation details

Current `qkt(...)` behavior:

- rejects null Q/K/score pointers or zero head dimension by returning without work;
- computes default scale as `1 / sqrt(head_dim)` when supplied scale equals zero;
- otherwise uses caller-provided scale;
- walks Q rows as tightly packed `head_dim` floats;
- walks K rows as tightly packed `head_dim` floats;
- computes each dot product with scalar F32 multiply-add accumulation;
- stores `dot * scale` into a tightly packed `[q_tokens, kv_tokens]` score matrix.

Native/library operations used include:

- pointer arithmetic;
- scalar multiplication/addition;
- `std::sqrt`;
- nested `std::size_t` loops.

## Mask implementation details

Current `apply_mask(...)` behavior:

- accepts a writable F32 score matrix;
- optional additive mask is also interpreted as tightly packed F32 with the same `[q_tokens, kv_tokens]` indexing;
- causal masking takes precedence: when `ki > qi`, score is replaced by negative infinity and additive mask processing for that element is skipped;
- otherwise an existing additive-mask value is added directly to the score.

Native/library operations used include:

- `std::numeric_limits<float>::infinity()`;
- scalar comparisons;
- pointer/index arithmetic;
- direct F32 add/store.

## Softmax implementation details

Current `softmax(...)` is a materialized, row-wise stable softmax.

For each row it:

1. finds the maximum score using `std::max`;
2. replaces each value with `exp(value - max)`;
3. accumulates the exponentials into an F32 sum;
4. if the sum is zero or non-finite, clears the whole row to zero;
5. otherwise multiplies every element by `1 / sum`.

Native/library operations used include:

- `std::numeric_limits<float>::infinity()`;
- `std::max`;
- `std::exp`;
- `std::isfinite`;
- scalar reciprocal/division;
- sequential row/column loops.

Important semantic fact: this differs from CUDA's current online softmax, especially in when special values are detected and when normalization occurs.

## AV implementation details

Current `av(...)` behavior:

- treats attention weights as tightly packed `[q_tokens, kv_tokens]` F32;
- treats V as tightly packed `[kv_tokens, head_dim]` F32;
- zero-initializes each output row;
- for each key/value token, loads one scalar attention weight;
- accumulates `weight * V[ki,d]` into every output dimension;
- output is tightly packed F32 `[q_tokens, head_dim]`.

Native operations are scalar pointer arithmetic, multiply, add, initialization, and nested loops.

## Temporary allocation and lifetime

`forward(...)` allocates score storage with `std::vector<float>`.

Relevant CPU-native behavior to preserve for later comparison:

- allocation occurs on the host heap through the C++ container;
- score memory lifetime is scoped to the `forward(...)` call;
- allocation size grows as `q_tokens * kv_tokens`;
- no tiling/streaming is used to bound score-memory growth;
- allocation failure would follow normal C++ allocation behavior rather than a GPU runtime error path.

## Layout assumptions

The current CPU implementation assumes contiguous row-major F32 buffers and does not accept explicit byte strides.

Assumptions include:

- Q: `[q_tokens, head_dim]` contiguous F32;
- K: `[kv_tokens, head_dim]` contiguous F32;
- V: `[kv_tokens, head_dim]` contiguous F32;
- mask: optional `[q_tokens, kv_tokens]` contiguous F32;
- scores: `[q_tokens, kv_tokens]` contiguous F32;
- output: `[q_tokens, head_dim]` contiguous F32.

Current CPU code does not encode batched/head dimensions separately; those would need to be represented by a higher-level caller, new interface, or expanded implementation.

## Dtype behavior

Current implementation is F32-only:

- Q: F32;
- K: F32;
- V: F32;
- additive mask: F32;
- scores: F32;
- accumulators: F32;
- output: F32.

There is no current F16/BF16 input conversion path, mixed-precision accumulation mode, quantized path, or vector packed dtype path.

## Synchronization and parallelism

The currently inspected implementation is sequential and therefore has no explicit synchronization primitive.

If later CPU implementations introduce concurrency, inventory those implementations separately. Possible categories include:

- `std::thread` / `std::jthread`;
- project thread-pool APIs;
- OpenMP pragmas/runtime calls;
- TBB/task runtimes;
- barriers, mutexes, atomics, latches, or condition variables;
- NUMA-aware partitioning.

Do not retroactively treat any of those as properties of the baseline CPU implementation.

## SIMD/vectorization categories

No explicit SIMD intrinsics are present in the current CPU translation.

If introduced later, inventory exact calls/instruction families separately, for example:

- x86 SSE/AVX/AVX2/AVX-512 intrinsics;
- AVX-512 BF16/FP16 or VNNI where relevant;
- ARM NEON/SVE/SVE2;
- compiler vector extensions;
- BLAS/vector-library calls.

Compiler auto-vectorization is an optimization property, not an explicit native call, unless build flags or assumptions make it part of the supported implementation contract.

## BLAS/library acceleration categories

Current code does not call BLAS.

If later used, record exact APIs and layouts, such as:

- CBLAS GEMM/GEMV;
- oneDNN primitives;
- Accelerate/vecLib;
- architecture-specific math libraries.

These could replace QKT/AV loops without changing broad attention semantics, but they must be inventoried as actual implementation choices.

## Math and special-value behavior

Current CPU path uses:

- `std::sqrt` for default scaling;
- negative infinity for causal-mask suppression;
- `std::max` for row maximum;
- `std::exp` for softmax exponentials;
- `std::isfinite` to reject zero/non-finite normalization sums.

Important special-case behavior:

- a fully masked row can produce `-inf - -inf`, yielding non-finite exponentiation behavior, after which the non-finite sum path clears the row to zero;
- the implementation does not have CUDA's explicit multiple-`+inf` online-softmax handling;
- NaN propagation is handled indirectly through the final non-finite sum test rather than by a dedicated per-score state machine.

These differences belong in later semantic comparison.

## Mask/bias features currently absent

The baseline CPU path has:

- additive F32 mask;
- optional simple causal mask.

It does not currently express:

- ALiBi slope logic;
- logit soft-cap;
- attention sinks;
- mask head/batch broadcasting;
- separate Q/K/V head counts;
- batch remapping;
- byte-strided tensor layouts.

Absence is part of the raw inventory and must not be hidden by later comparison.

## Error/status behavior

Current functions generally fail by early return rather than structured error objects.

`forward(...)` returns `false` when:

- Q/K/V/output is null;
- query-token count is zero;
- key/value-token count is zero;
- head dimension is zero.

Sub-functions return `void` and silently perform no work for selected invalid-pointer/dimension cases.

There is no current exception-to-status normalization, diagnostic string, backend error code, or runtime device error.

## Current performance characteristics to preserve as facts

- scalar F32 arithmetic;
- O(Q*K*D) QKT work;
- O(Q*K) score storage;
- O(Q*K) mask and softmax work;
- O(Q*K*D) AV work;
- multiple passes over the materialized score matrix;
- no explicit parallelism;
- no explicit SIMD;
- no fusion between QKT, masking, softmax, and AV.

These are baseline implementation facts, not requirements of the future CPU backend.

## Current repository symbols to preserve in later comparison

- `easyapi::attention::xformers::cpu::qkt(...)`
- `easyapi::attention::xformers::cpu::apply_mask(...)`
- `easyapi::attention::xformers::cpu::softmax(...)`
- `easyapi::attention::xformers::cpu::av(...)`
- `easyapi::attention::xformers::cpu::forward(...)`

## Inventory gaps to fill from future CPU implementations

- batched/head-aware interfaces;
- strided tensor support;
- F16/BF16 input paths;
- SIMD implementation;
- multithreading/thread-pool implementation;
- tiled/streaming score handling;
- online softmax;
- ALiBi;
- logit soft-cap;
- attention sinks;
- architecture-specific optimized kernels;
- BLAS/library fallback paths;
- deterministic/reproducibility modes.

## Questions reserved for the later comparison phase

Do not answer these here.

- Is the staged CPU sequence a semantic requirement or merely the current baseline implementation?
- Which CPU operations correspond to indivisible native operations on GPU backends?
- Should temporary score allocation be visible to translation or completely backend-private?
- Which special-value differences must be normalized to achieve equivalent behavior?
- Should dtype conversion and layout adaptation be separate translated concepts or implementation details?
