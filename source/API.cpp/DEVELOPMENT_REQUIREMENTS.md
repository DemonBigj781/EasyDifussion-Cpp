# API.cpp Development Requirements

## Purpose

API.cpp is a self-contained hardware unifier. It presents one API-owned context
and one normalized set of tensor, memory, model-resource, and attention
operations across supported hardware backends.

API.cpp primarily manages and unifies resources. It may also implement inference
operations, but those implementations must be owned by API.cpp and use its own
context. It must never delegate inference to SDKIT3 or llama.cpp.

## Dependency direction

The only permitted in-repository consumer of API.cpp is `source/API.test`:

```text
API.test -> API.cpp -> platform and driver interfaces
```

The reverse direction is forbidden. API.cpp must build and remain usable when
`API.test`, SDKIT3, and llama.cpp are absent.

SDKIT3 and llama.cpp must not include, compile, link, dispatch, configure, or
otherwise consume API.cpp. API.cpp must not consume either engine. They are
research/reference options only until the project owner explicitly changes
their status. `API.test` may depend on API.cpp, SDKIT3, and llama.cpp when an
integration test requires all three.

## Owned context and implementations

- API.cpp defines and owns every public context, handle, tensor descriptor,
  capability record, request, result, and lifecycle contract it exposes.
- Backend translations consume API-owned normalized requests and translate them
  directly to the selected driver interface.
- Backend implementations must live inside API.cpp. Wrapping or importing an
  inference engine's context or implementation does not satisfy this rule.
- SDKIT3 is not a source donor. When an implementation already exists in
  API.cpp, API routes use the API-owned implementation. When it does not exist,
  the route remains unimplemented until it is independently developed against
  the API and driver contracts; code must never be pulled from SDKIT3 to fill it.
- The same rule applies to llama.cpp. Research may inform requirements and test
  cases, but production implementation code must be independently owned by
  API.cpp rather than copied, wrapped, or dispatched from either research tree.
- Opaque driver handles are permitted when their meaning is documented by the
  corresponding backend translation. Opaque inference-engine handles are not.
- Model loading and unloading mean resource residency and lifecycle management;
  they do not imply model-specific inference execution.

## Permitted external dependencies

Production API.cpp code may use:

- the C and C++ standard libraries;
- operating-system interfaces required for devices, memory, files, and dynamic
  driver discovery;
- vendor or platform driver/runtime interfaces such as CUDA, HIP/ROCm, OpenCL,
  Vulkan, oneAPI/Level Zero, OpenVINO device interfaces, DirectML, and equivalent
  hardware APIs.

Inference engines, inference frameworks, application SDKs, and their private
tensor or scheduler contexts are not permitted dependencies. In particular,
API.cpp must not depend on SDKIT3, stable-diffusion.cpp, GGML, or llama.cpp.

## Attention requirements

- Common owns the normalized attention request and capability contract.
- Definition describes what a backend implementation provides.
- Translation maps the Common request to an API-owned backend implementation
  and, where needed, directly to its driver runtime.
- CPU, Flex Attention, and Split Attention must use API-owned data structures.
- A source or prototype route is not reported as build-ready or runtime-verified
  until its corresponding checks pass.

## Enforcement

`scripts/test_api_boundaries.py` rejects production cross-library references,
including any API.cpp dependency on `API.test`. Build workflows must compile
production API.cpp routes using API.cpp and driver include paths only. Engine
include paths are allowed only inside explicit `API.test` integration builds.
