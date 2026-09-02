# Common routing audit

This document records the architectural expectation that every feature speaks one normalized Common language to the Library and application.

## Required route

```text
Application / Easy Diffusion
    -> Library
    -> feature Common API
    -> backend translation
    -> backend definition
    -> native runtime/device
```

Results return through the reverse route. The Library must never call a backend translation or definition directly.

## Validation dimensions

Behavioral proof and architectural proof are separate.

- **Behaviorally proven** means the tested implementation executed correctly for the recorded scenario.
- **Common contract present** means a non-placeholder feature-specific Common API owns normalized request/result semantics.
- **Common-only routing** means the Library reaches the backend only through Common; no Library source includes backend translation/definition headers.
- **Architecturally proven** means the complete Common-only route has been compiled, runtime-tested, and exercised end to end.

Existing behavioral test results are retained when a routing defect is found. Fixing the route does not erase the earlier test history, but architectural completion must be re-proven through Common.

## Current audit of previously tested features

### Detect

Feature-specific Common implementation exists. CPU and CUDA Library handlers now call the Common Detect entry point only. CPU/CUDA translations self-register with Common and remain responsible for reaching their backend definitions. This migration is coded but must be compiled and runtime-revalidated before Common-only routing is considered architecturally proven.

### Load / model lifecycle

Feature-specific Common implementation exists. CPU and CUDA Library model lifecycle handlers now call Common `load_model` only. CPU/CUDA model-load translations self-register with Common. The earlier behavioral proof remains valid, but the new Common-only route requires compile and runtime revalidation.

### Unload / model lifecycle

Feature-specific Common implementation exists. CPU and CUDA Library model lifecycle handlers now call Common `unload_model` only. Resource ownership selects the registered backend translation inside Common. The earlier behavioral proof remains valid, but the new Common-only route requires compile and runtime revalidation.

### Overflow

Feature-specific Common planning, normalization, and normalized resource contracts exist. Overflow is intentionally deferred because its policy and execution model are substantially more complex and only partially planned beyond the proven CPU/CUDA paths. Existing CPU/CUDA overflow runtime and stress-test results remain behavioral evidence; no Common-routing refactor is claimed here yet.

### xFormers

The root Common API already knows the `xformers_attention` operation/capability, but the feature-specific xFormers Common method files were originally structural placeholders. xFormers must establish its authoritative Common contract and route Library calls through it before backend implementations are considered integrated.

## Development expectation

For new or unfinished features:

1. Establish the normalized feature Common contract.
2. Define backend-native semantics in `definition/`.
3. Implement backend adapters in `translation/` against the Common contract.
4. Register translations with Common or otherwise connect them through a backend-neutral Common dispatch mechanism.
5. Connect Library only to Common.
6. Compile the route.
7. Runtime-test backend behavior.
8. End-to-end test Library -> Common -> translation -> definition -> native execution and the normalized return route.
9. Claim architectural completion only after the Common-only route and required feature behavior are thoroughly accounted for.

## Enforcement target

A future source/CI audit should fail when a file under `source/API.cpp/library/` includes a path containing `/translation/` or `/definition/`. Equivalent direct backend dependencies should also be rejected during review even if expressed through an alias or wrapper.
