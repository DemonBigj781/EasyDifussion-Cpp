# CUDA — EasyCache

## Status
Native cache policy, backend-neutral host implementation.

## Implementation
`src/runtime/easycache.hpp` implements EasyCache in C++. It tracks prior diffusion inputs/outputs, estimates the relative transformation rate, accumulates predicted output change, and reuses cached condition diffs while the cumulative change remains below the configured threshold.

## Cache placement
The current cache state is host-resident. Cached condition diffs, previous inputs, and previous outputs are stored in `std::vector<float>` containers. This page therefore does not claim a CUDA-resident cache kernel or VRAM-native cache store.

## CUDA relationship
CUDA executes the underlying denoiser/model work. EasyCache decides whether that work can be skipped and reconstructs the condition output from the stored host-side diff. Generic EasyCache policy should remain outside CUDA-specific backend code unless a future CUDA implementation introduces device-resident state, kernels, synchronization, or capability handling.

## Configuration
Current configuration includes `enabled`, `reuse_threshold`, `start_percent`, and `end_percent`. The active range is converted to sigma through the current denoiser.

## Invalidation / reset
Runtime state is cleared by `reset_runtime()`, including cached diffs, previous tensors, accumulated rates, anchor condition, and skipped-step count.

## Fallback
When EasyCache is disabled, outside its active sigma window, lacks sufficient previous state, or cannot safely reuse a condition, the normal model evaluation proceeds.

## Validation
Native policy implementation is present. CUDA-specific performance and transfer-cost validation remain required; no CUDA-specific EasyCache kernel exists at this stage.
