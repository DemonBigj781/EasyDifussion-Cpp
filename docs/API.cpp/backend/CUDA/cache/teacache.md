# CUDA — TeaCache

## Status
Native cache policy, backend-neutral host implementation.

## Implementation
`src/runtime/teacache.hpp` implements the TeaCache policy in C++. It measures relative L1 change between diffusion inputs, applies model-specific polynomial rescaling, accumulates that change, and reuses a previously stored full-model residual while the accumulated value remains below the configured threshold.

## Cache placement
The current TeaCache state is explicitly host-resident. Previous diffusion inputs and cached residuals are stored in `std::vector<float>` containers. The implementation intentionally keeps the residual in host memory and performs its probe at the condition boundary so streamed weights do not require a second partial graph.

## CUDA relationship
CUDA executes the underlying model. TeaCache determines whether a step can reuse the host-stored residual. This is not currently a CUDA-resident cache implementation and does not claim a dedicated CUDA TeaCache kernel.

## Configuration
Current configuration includes `enabled`, `reuse_threshold`, `start_percent`, `end_percent`, `total_steps`, model-specific polynomial `coefficients`, and `model_variant`.

## Fallback
The normal model evaluation proceeds when TeaCache is disabled, outside the active step range, on the first or last step, when tensor sizes change, when no residual exists, or when accumulated change exceeds the reuse threshold.

## Validation
Native policy implementation is present. CUDA-specific performance, host/device transfer cost, and model-quality validation remain required.
