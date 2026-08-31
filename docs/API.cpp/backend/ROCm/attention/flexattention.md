# ROCm — FlexAttention token selection

## Status
Native backend-neutral host implementation; no ROCm-specific kernel currently required.

## Scope
This is the UMass-Embodied-AGI FlexAttention visual-token-selection policy, not PyTorch's unrelated generic FlexAttention API.

## Implementation
The C++ selector in `src/runtime/flex_attention.hpp` consumes low-resolution attention values, combines heads, normalizes and thresholds query maps, adaptive-max-pools them, and expands the selection mask to the high-resolution vision grid.

## Backend relationship
ROCm/HIP is responsible for producing the attention values used by the model. The current FlexAttention selection policy itself executes in backend-neutral host C++.

## Fallback
Invalid input/configuration produces an empty selection result.

## Validation
Algorithm implementation is present. End-to-end ROCm integration and performance remain separate validation tasks.
