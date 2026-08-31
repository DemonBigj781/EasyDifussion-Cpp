# CPU — FlexAttention token selection

## Status
Native backend-neutral host implementation.

## Scope
This project uses the UMass-Embodied-AGI FlexAttention visual-token-selection policy, not PyTorch's generic FlexAttention API.

## Implementation
`src/runtime/flex_attention.hpp` performs the token-selection stage entirely in C++ using host data structures and arithmetic: head summation, normalization, thresholding, adaptive max pooling, and expansion to the high-resolution vision grid.

## CPU relationship
The current selection implementation is naturally executable on CPU and does not require a GPU backend.

## Fallback
Invalid inputs or configuration return an empty selection result.

## Validation
Algorithm implementation is present. End-to-end model integration should be validated independently from the selector implementation.
