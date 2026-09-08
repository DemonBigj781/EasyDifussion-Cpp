# CPU — TeaCache

## Status
Native.

## Implementation
TeaCache is implemented directly in host C++ in `src/runtime/teacache.hpp`. It stores previous diffusion inputs and the reusable residual in host `std::vector<float>` containers, measures relative L1 change, applies model-specific polynomial rescaling, and skips model evaluation while accumulated change remains below the configured threshold.

## CPU relationship
The current TeaCache implementation is inherently host-side and requires no GPU-specific cache kernel for CPU execution.

## Fallback
Normal model evaluation proceeds when reuse is disabled, inactive, unsafe, or above threshold.

## Validation
Implementation is present. Model-quality, threshold, polynomial-coefficient, and performance validation remain required across supported model variants.
