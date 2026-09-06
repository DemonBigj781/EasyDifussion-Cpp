# Common API

Backend-neutral contracts for `source/API.cpp` live here. Backend implementations
must depend on these types, while callers use this layer instead of directly
selecting CPU, CUDA, ROCm, oneAPI, OpenCL, OpenVINO, OpenGL, Vulkan, Mesa, or
DirectML code.

## Architectural invariant

**Common is the only language shared with the Library and application.**

The required downward execution route is:

```text
Easy Diffusion / application
    -> feature Common exported API
    -> backend translation
    -> backend definition
    -> native backend/runtime/device
```

The normalized result returns through the reverse route:

```text
native backend/runtime/device
    -> backend definition
    -> backend translation
    -> feature Common exported API
    -> Easy Diffusion / application
```

`library/` may package the Common ABI, but it is not a runtime routing layer and
must not include or call a backend definition or translation directly.
Backend-specific selection, native objects, kernels, SDK calls, and workarounds
remain below Common. Common owns the normalized request and result language
used by both the application and any Library facade.

## Files

- `api.hpp` / `api.cpp`: backend identifiers, operation identifiers, capability flags, device metadata, dispatch context, and name helpers.
- `registry.hpp` / `registry.cpp`: fixed-size backend registration, capability lookup, and dispatch routing.
- Feature-specific common contracts live under `features/[family]/[feature]/common/`;
  this root directory retains only library-wide registry and dispatch contracts.

## Rules

- No backend-specific headers or kernels belong in `common/`.
- No `library/` source may include a path below a backend `definition/` or `translation/` directory.
- Backend implementations register one `BackendInterface` with the common registry or are reached through an equivalent feature-Common dispatch contract.
- A backend advertises only capabilities it can actually satisfy.
- Feature Common APIs define normalized inputs, outputs, errors, ownership, capability checks, and dispatch behavior before the Library treats that feature as integrated.
- Flash, Sage, and xFormers compatibility are represented as common operations/capabilities; each feature translation adapts its backend to that Common contract.
- Backend-specific behavior may differ internally, but translation must make equivalent behavior speak the same Common language.
- Compile success, backend runtime success, and Common-only architectural integration are separate validation states.
- `99 - API` can compile this layer in isolation, while normal backend workflows compile it as part of the full project.
