# ROCm — EasyCache

## Status
Native cache policy, backend-neutral host implementation.

## Implementation
EasyCache is implemented in `src/runtime/easycache.hpp`. It tracks prior diffusion inputs and outputs, estimates transformation rate, accumulates predicted change, and reuses stored condition diffs while reuse remains under the configured threshold.

## Cache placement
Cache diffs and prior tensor snapshots are stored in host `std::vector<float>` containers. No ROCm/HIP-resident EasyCache state or dedicated HIP cache kernel is currently present.

## ROCm relationship
ROCm executes the underlying denoiser/model work. EasyCache is a host-side decision and reconstruction layer around that work.

## Fallback
Normal model evaluation proceeds whenever the cache is disabled, inactive, uninitialized, missing required history, or exceeds its reuse threshold.

## Validation
Policy implementation is present. ROCm-specific performance and host/device transfer-cost validation remain required.
