# CPU — EasyCache

## Status
Native.

## Implementation
EasyCache is implemented directly in host C++ in `src/runtime/easycache.hpp`. It stores prior diffusion inputs/outputs and condition diffs in `std::vector<float>` containers, estimates relative transformation behavior, and reuses cached diffs when the configured reuse threshold allows it.

## CPU relationship
Because the cache state and policy are already host-resident, CPU execution does not require a backend bridge for the core EasyCache mechanism.

## Fallback
Normal model evaluation proceeds whenever cache reuse is not valid or active.

## Validation
Implementation is present. Model-quality and performance validation should cover cache-hit behavior, threshold tuning, and supported model families.
