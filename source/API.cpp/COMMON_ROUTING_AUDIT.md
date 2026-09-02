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

## Initial audit of previously tested features

### Detect

Feature-specific Common implementation exists. Current Library handler code has been observed including and calling backend translation directly before Common normalization. Behavioral proof remains valid; Common-only routing still requires migration and revalidation.

### Load / model lifecycle

Feature-specific Common implementation exists. Current Library model lifecycle code has been observed including backend-specific load translation directly and then passing its result through Common normalization. Behavioral proof remains valid; Common-only routing still requires migration and revalidation.

### Unload / model lifecycle

Feature-specific Common implementation exists. Current Library model lifecycle code has been observed including backend-specific unload translation directly and then passing its result through Common normalization. Behavioral proof remains valid; Common-only routing still requires migration and revalidation.

### Overflow

Feature-specific Common planning, normalization, and normalized resource contracts exist. Library/backend overflow paths must be audited for the same Common-only rule before architectural completion is claimed. Existing CPU/CUDA overflow runtime and stress-test results remain behavioral evidence.

### xFormers

The root Common API already knows the `xformers_attention` operation/capability, but the feature-specific xFormers Common method files were originally structural placeholders. xFormers must establish its authoritative Common contract and route Library calls through it before backend implementations are considered integrated.

## Development expectation

For new or unfinished features:

1. Establish the normalized feature Common contract.
2. Define backend-native semantics in `definition/`.
3. Implement backend adapters in `translation/` against the Common contract.
4. Connect Common dispatch to the translations without exposing backend-native objects upward.
5. Connect Library only to Common.
6. Compile the route.
7. Runtime-test backend behavior.
8. End-to-end test Library -> Common -> translation -> definition -> native execution and the normalized return route.
9. Claim architectural completion only after the Common-only route and required feature behavior are thoroughly accounted for.

## Enforcement target

A future source/CI audit should fail when a file under `source/API.cpp/library/` includes a path containing `/translation/` or `/definition/`. Equivalent direct backend dependencies should also be rejected during review even if expressed through an alias or wrapper.
