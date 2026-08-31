# Common API

Backend-neutral contracts for `source/API.cpp` live here. Backend implementations must depend on these types, while callers use this layer instead of directly selecting CUDA, ROCm, oneAPI, Vulkan, or CPU code.

## Files

- `api.hpp` / `api.cpp`: backend identifiers, operation identifiers, capability flags, device metadata, dispatch context, and name helpers.
- `registry.hpp` / `registry.cpp`: fixed-size backend registration, capability lookup, and dispatch routing.
- `attention/`: the shared optimized-attention contract and backend-neutral host FlexAttention implementation.

## Rules

- No backend-specific headers or kernels belong in `common/`.
- Backend implementations register one `BackendInterface` with the common registry.
- A backend advertises only capabilities it can actually satisfy.
- Flash, Sage, and xFormers compatibility are represented as common operations/capabilities; each backend decides how those operations are implemented.
- `99 - API` can compile this layer in isolation, while normal backend workflows compile it as part of the full project.
