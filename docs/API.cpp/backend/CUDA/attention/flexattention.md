# CUDA — FlexAttention token selection

## Status
Native backend-neutral host implementation; no CUDA-specific kernel currently required.

## Scope
This project uses the UMass-Embodied-AGI FlexAttention visual-token-selection policy. It is not documentation for PyTorch's unrelated generic FlexAttention API.

## Implementation
`src/runtime/flex_attention.hpp` implements the inference-time selection stage in C++. It sums low-resolution visual attention across heads, normalizes each query map, thresholds in 8-bit space, adaptive-max-pools to a 9x9 grid, and expands the result to the high-resolution vision grid.

## Backend relationship
The selection policy operates on attention values exposed to host C++ code. CUDA remains responsible for the underlying model/attention execution; the current FlexAttention selection step itself is backend-neutral.

## Requirements
The caller must provide valid attention data, positive head/query counts, and a valid `SelectionConfig`.

## Fallback
If the configuration or input is invalid, the selector returns an empty result rather than invoking a backend-specific fallback.

## Validation
Algorithm implementation is present. End-to-end CUDA integration and performance should be validated separately from algorithm correctness.
