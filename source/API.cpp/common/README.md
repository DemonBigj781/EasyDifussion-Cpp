# Common API

Backend-neutral contracts for `source/API.cpp` live here. Backend implementations
must depend on these types, while callers use this layer instead of directly
selecting CPU, CUDA, ROCm, oneAPI, OpenCL, OpenVINO, OpenGL, Vulkan, Mesa, or
DirectML code.

## Files

- `api.hpp` / `api.cpp`: backend identifiers, operation identifiers, capability flags, device metadata, dispatch context, and name helpers.
- `registry.hpp` / `registry.cpp`: fixed-size backend registration, capability lookup, and dispatch routing.
- Feature-specific common contracts live under `features/[family]/[feature]/common/`;
  this root directory retains only library-wide registry and dispatch contracts.

## Rules

- No backend-specific headers or kernels belong in `common/`.
- Backend implementations register one `BackendInterface` with the common registry.
- A backend advertises only capabilities it can actually satisfy.
- Flash, Sage, and xFormers compatibility are represented as common operations/capabilities; each feature translation adapts its backend before registration here.
- `99 - API` can compile this layer in isolation, while normal backend workflows compile it as part of the full project.
