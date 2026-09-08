# DiffUser.cpp Development Requirements

## Purpose

DiffUser.cpp is a self-contained hardware unifier. It presents one API-owned context
and one normalized set of tensor, memory, model-resource, and attention
operations across supported hardware backends.

DiffUser.cpp manages and unifies resources and backend compute primitives. It
owns the administrative path: discovery, capability reporting, allocation,
placement, transfer, model-resource load/unload, dispatch bookkeeping, and
release. Model-family inference, sampling, VAE encode/decode semantics, and
delivery of completed inference results belong to INFERENCE.cpp. Neither
project delegates implementation to SDKIT3 or llama.cpp.

The handoff rule is: DiffUser prepares the work; INFERENCE uses the prepared
work. A prepared work package may expose Common-owned resource handles, tensor
descriptions, placement, synchronization, and supported primitive operations.
It must not contain model-family control flow, sampler policy, VAE algorithm
policy, or result-delivery decisions.

Common is the final normalization layer. It exposes the same operation and
resource semantics regardless of which backend, device class, definition, or
translation produced them. Callers must never select a different API because a
resource originated on a different backend.

Backend definitions may use their driver's native types and calling conventions
internally. At every boundary outside that backend/device implementation, the
translation must convert those native values to the Common contract. Backends
and devices must not call one another's native interfaces directly: all
cross-device coordination and all higher-level access use Common as their one
shared language.

The same rule applies when an operation begins and ends on the same device.
Same-device placement is not permission for a caller to bypass Common. Native
driver language exists only during unobserved work inside one definition. Any
message entering or leaving that definition is translated: the route is
`native -> translation -> Common -> translation -> native`, even for a
self-targeted operation. There is no native-to-native device shortcut and no
Common-to-Common translation hop; Common is the shared message contract, not a
device implementation.

## Dependency direction

`source/API.cpp` is a compatibility filesystem link to the canonical
`source/DiffUser.cpp` tree. It must never become a separate implementation.

The permitted in-repository consumers of DiffUser.cpp Common are
`source/INFERENCE.cpp` and `source/API.test`:

```text
INFERENCE.cpp -> DiffUser Common -> translation -> definition -> driver
API.test      -> DiffUser Common -> translation -> definition -> driver
```

The reverse direction is forbidden. DiffUser.cpp must build and remain usable when
`API.test`, SDKIT3, and llama.cpp are absent.

INFERENCE.cpp must consume only Common/public contracts. It must not include a
backend definition, translation, handler, or driver surface. If an inference
operation lacks a required normalized Common contract, that contract is added
to DiffUser.cpp before INFERENCE.cpp uses the operation. Every hardware-backed
contract must also have definitions and translations for each backend claimed
as supported, including the applicable GPU translations. A Common header alone
does not establish a usable operation.

The dated `Audit/` archive tracks INFERENCE.cpp-to-Common gaps. Each inference
feature migration must update that audit before implementation so missing
contracts and GPU translations are discovered before a private substitute is
introduced into INFERENCE.cpp.

VAE encode/decode algorithms and their sequencing belong to INFERENCE.cpp.
DiffUser owns only the normalized VAE model, image, latent, placement, transfer,
and compute handlers required by that implementation, together with their
backend definitions and translations.

SDKIT3 and llama.cpp must not include, compile, link, dispatch, configure, or
otherwise consume DiffUser.cpp. DiffUser.cpp must not consume either engine. They are
research/reference options only until the project owner explicitly changes
their status. `API.test` may depend on DiffUser.cpp, SDKIT3, and llama.cpp when an
integration test requires all three.

## Owned context and implementations

- DiffUser.cpp defines and owns every public context, handle, tensor descriptor,
  capability record, request, result, and lifecycle contract it exposes.
- Backend translations consume API-owned normalized requests and translate them
  directly to the selected driver interface.
- Backend resource and compute-primitive implementations must live inside
  DiffUser.cpp. Model-level inference implementations must live inside
  INFERENCE.cpp. Wrapping or importing another inference engine's context or
  implementation does not satisfy either rule.
- SDKIT3 is not a source donor. When an implementation already exists in
  DiffUser.cpp, API routes use the API-owned implementation. When it does not exist,
  the route remains unimplemented until it is independently developed against
  the API and driver contracts; code must never be pulled from SDKIT3 to fill it.
- The same rule applies to llama.cpp. Research may inform requirements and test
  cases, but production implementation code must be independently owned by
  DiffUser.cpp rather than copied, wrapped, or dispatched from either research tree.
- Opaque driver handles are permitted when their meaning is documented by the
  corresponding backend translation. Opaque inference-engine handles are not.
- Model loading and unloading mean resource residency and lifecycle management;
  they do not imply model-specific inference execution.

## Permitted external dependencies

Production DiffUser.cpp code may use:

- the C and C++ standard libraries;
- operating-system interfaces required for devices, memory, files, and dynamic
  driver discovery;
- vendor or platform driver/runtime interfaces such as CUDA, HIP/ROCm, OpenCL,
  Vulkan, oneAPI/Level Zero, OpenVINO device interfaces, DirectML, and equivalent
  hardware APIs.

Inference engines, inference frameworks, application SDKs, and their private
tensor or scheduler contexts are not permitted dependencies. In particular,
DiffUser.cpp must not depend on SDKIT3, stable-diffusion.cpp, GGML, or llama.cpp.

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
including any DiffUser.cpp dependency on `API.test`. Build workflows must compile
production DiffUser.cpp routes using DiffUser.cpp and driver include paths only. Engine
include paths are allowed only inside explicit `API.test` integration builds.
