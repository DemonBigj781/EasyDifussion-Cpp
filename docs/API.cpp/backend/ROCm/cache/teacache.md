# ROCm — TeaCache

## Status
Native cache policy, backend-neutral host implementation.

## Implementation
TeaCache is implemented in `src/runtime/teacache.hpp`. It measures relative L1 change between diffusion inputs, applies model-specific polynomial rescaling, accumulates that change, and reuses a stored residual while the accumulated value remains below the configured threshold.

## Cache placement
The current residual and prior diffusion input are host-resident `std::vector<float>` data. No ROCm/HIP-resident TeaCache state or dedicated HIP cache kernel is currently present.

## ROCm relationship
ROCm executes the underlying model. TeaCache determines whether that execution can be skipped and whether the host-stored residual can be reused.

## Fallback
Normal model evaluation proceeds on disabled/inactive cache states, first and last steps, shape changes, missing residuals, or threshold exceedance.

## Validation
Policy implementation is present. ROCm-specific transfer overhead, quality, and performance validation remain required.
