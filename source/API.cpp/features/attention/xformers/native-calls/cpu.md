# CPU native-operation inventory for xFormers

Status: raw inventory. Do not choose normalized translation names from this file yet.

The CPU path is not a GPU API, so this inventory records the host-language operations and execution structure actually used by the current xFormers CPU translation.

## Execution model

- Ordinary C++ function calls.
- Sequential orchestration in `forward(...)`.
- `std::vector<float>` for temporary score storage.
- Raw pointer inputs/outputs for Q, K, V, mask, scores, and output buffers.

## Current logical execution sequence

The current CPU `forward(...)` explicitly performs:

1. `qkt(...)`
2. `apply_mask(...)`
3. `softmax(...)`
4. `av(...)`

Unlike the CUDA implementation, the CPU path materializes the score matrix in a temporary `std::vector<float>` before mask/softmax/AV processing.

## Current public/internal operations

- `qkt(const float*, const float*, float*, ...)`
- `apply_mask(float*, const float*, ..., bool causal)`
- `softmax(float*, ...)`
- `av(const float*, const float*, float*, ...)`
- `forward(...)`

## Synchronization and parallelism

The currently inspected CPU `forward(...)` does not expose an explicit thread-group/barrier abstraction. Any later SIMD, OpenMP, thread-pool, or vector-intrinsic implementation must be inventoried separately rather than treated as implicit behavior of this baseline.

## Dtype / layout

The current CPU forward interface is float-based. It does not presently express the CUDA path's F16/BF16 dispatch, stream object, GPU shared memory, block geometry, or fused online-softmax execution.

## Important comparison fact for later

CPU's staged implementation and CUDA's fused streaming implementation are different execution strategies for the same broad attention result. This is exactly why the later translation vocabulary must be derived from semantic requirements rather than from one backend's physical stage boundaries.

## Not yet claimed

This document does not define which CPU functions become normalized translation calls. It only records the current implementation shape.
