# Common routing audit

This document records the 2026-09-06 re-audit after upstream Theory commit
`14c4542` and the current layout changes. Every feature speaks one normalized
Common language to the application.

Directory status and superseded/future paths are tracked separately in
`LAYOUT_AUDIT.md`.

## Required route

```text
Application / Easy Diffusion
    -> feature Common API
    -> backend translation
    -> backend definition
    -> native runtime/device
```

Results return through the reverse route. Only Common may communicate with the
application. A backend communicates with Common through its translation and
must never call the program directly. `library/` is optional export/ABI
packaging around Common, not a routing layer; it must never call a backend
translation or definition directly.

## Validation dimensions

Behavioral proof and architectural proof are separate.

- **Behaviorally proven** means the tested implementation executed correctly for the recorded scenario.
- **Common contract present** means a non-placeholder feature-specific Common API owns normalized request/result semantics.
- **Common-only routing** means the Library reaches the backend only through Common; no Library source includes backend translation/definition headers.
- **Architecturally proven** means the complete Common-only route has been compiled, runtime-tested, and exercised end to end.

Existing behavioral test results are retained when a routing defect is found. Fixing the route does not erase the earlier test history, but architectural completion must be re-proven through Common.

## Current audit of previously tested features

### Detect

All Library handlers call the Common Detect entry point only. Backend
translations register behind Common and remain responsible for reaching their
definitions. CPU, CUDA, OpenCL, OpenGL, Vulkan, and Mesa were rebuilt and
runtime-tested; Vulkan combined results were also verified to have globally
unique normalized indices. ROCm, oneAPI, OpenVINO, and DirectML remain
source-audited only because their local toolchains/runtimes were unavailable.

### Load / model lifecycle

CPU and CUDA Library model lifecycle handlers call Common `load_model` only,
and their translations self-register behind Common. Both complete routes were
rebuilt and runtime-tested.

### Unload / model lifecycle

CPU and CUDA Library model lifecycle handlers call Common `unload_model` only.
Resource ownership selects the registered translation inside Common. Both
routes, including mismatch and double-unload rejection, were rebuilt and
runtime-tested.

### Overflow

Common now owns Overflow planning, normalization, and registered allocation and
release dispatch. CPU tests passed. CUDA tests passed on an RTX 3060, including
forced Managed Memory fallback and a bounded real `cudaMalloc` OOM followed by
GPU access, CPU verification, and release. Non-CUDA backend overflow remains
unimplemented.

### xFormers

The Common contract owns normalized tensor metadata, dtypes/strides,
capabilities, validation, staged execution, fused-forward execution, and
translation registration. CPU registers staged F32 execution. CUDA registers
one fused callback supporting F32/F16/BF16 Q/K/V, F32 output, byte strides,
mask broadcasting/dtypes, head and batch broadcasting, ALiBi, soft-cap,
attention sinks, and an opaque CUDA stream.

The standalone cycle tests call Common exclusively and pass 32 custom model
load/forward/unload cycles; CUDA also passes Compute Sanitizer with zero
errors. The stable-diffusion.cpp GGML application adapter now converts into
that same Common request. A 512x512 one-step generation completed with 20
native CUDA xFormers launches. There is no separate production GGML-native
xFormers backend route.

### FlashAttention

The feature Common API owns a backend-neutral tensor, parameter, execution, capability, validation, and result contract. The CPU translation maps that contract to the existing GGML `FLASH_ATTN_EXT` compatibility definition; GGML tensor and compute objects do not cross the feature Common boundary. Workflow `001` compiles the route and runs a native-boundary smoke test covering registration, normalized validation, capability reporting, translation, and exactly one native dispatch.

CUDA has its own definition and registered `forward.cu` translation behind the
same Common contract. It uses fused online softmax without materializing the QK
matrix and supports F32/F16/BF16 inputs, F32 output, additive masks,
GGML-style max-bias/ALiBi mask scaling, logit soft-capping, grouped-query
attention, byte strides, and an opaque CUDA stream. A Common-only numerical
test passed on an RTX 3060 (`sm_86`, driver 580.94.18, CUDA 12.4), including
Compute Sanitizer memcheck with zero errors.

The CUDA Flash roundtrip MultiTest then completed 32 cycles in which model
bytes entered through Common load and returned unchanged, Flash input entered
Common and its result returned numerically correct, and the model resource
exited through Common unload. This proves repeated API roundtrip behavior; it
does not substitute for image-generation integration.

CPU remains Common-routed build/native-boundary smoke evidence rather than
numerical proof. CUDA is runtime-proven but not end-to-end: the optimized
stable-diffusion.cpp GGML `fattn` path remains a separate compatibility route
and no Easy Diffusion workflow enters Flash Common yet. The CPU translation
also accepts only single-thread execution until backend-native thread-pool
scheduling is normalized. The CUDA Common route requires Pascal or newer; the
Nouveau-driven Kepler K2000 is neither a CUDA runtime device nor eligible for
that route.

## Development expectation

For new or unfinished features:

1. Establish the normalized feature Common contract.
2. Define backend-native semantics in `definition/`.
3. Implement backend adapters in `translation/` against the Common contract.
4. Register translations with Common or otherwise connect them through a backend-neutral Common dispatch mechanism.
5. Connect Library only to Common.
6. Compile the route.
7. Runtime-test backend behavior.
8. End-to-end test application -> Common -> translation -> definition -> native execution and the normalized return route.
9. Claim architectural completion only after the Common-only route and required feature behavior are thoroughly accounted for.

## Enforcement target

A future source/CI audit should fail when a file under `source/API.cpp/library/` includes a path containing `/translation/` or `/definition/`. Equivalent direct backend dependencies should also be rejected during review even if expressed through an alias or wrapper.

The current `library/` source audit finds no translation/definition includes.
The removed `library/backends/` tree should remain absent. Remaining attention
routing debt is in the Flash/Sage GGML-native application compatibility paths,
not in Library handlers. Flash's separate CUDA Common translation does not by
itself retire that debt. Reserved Mesa/OpenGL/Vulkan lifecycle, Overflow, and
xFormers test paths have no implementation behind them.
