# GGML WebGPU through the static Cosmopolitan runtime

This adapter connects the existing shared GGML WebGPU backend to the pinned
wgpu-native C ABI and the embedded lavapipe/LLVM runtime. It preserves the
customized SD GGML as the authoritative implementation. The llama and SD
consumers use the same GGML registry, device, allocator and backend code.

`shared-ggml/patches/0004-wgpu-native-c-api-backend.patch` adapts that backend's
initialization and CMake dependency. It does not replace the WGSL kernels with
CPU tensor operations. The GGML C API remains `ggml_backend_webgpu_reg()` and
`ggml_backend_webgpu_init()`. The registry name is `WebGPU`, and its sole device
is `WebGPU0`.

`include/webgpu/webgpu_cpp.h` implements only the C++ API subset used by this
pinned GGML backend. Every resource and compute operation goes through the
actual wgpu-native C functions. Resources use shared ownership with one native
reference per owned handle. Descriptor conversion uses the exact pinned C
structures and initializer macros; it does not assume Dawn structure layout or
ABI compatibility. This is not a general implementation of Dawn's C++ API.

The foundation pin is owned by `../software-webgpu`. Its WebGPU API is
wgpu-native v29.0.1.1, revision `6aed50955d934ac36049ba8d002034841633ae02`,
with webgpu-headers revision `673658bc2bd70ec39fc55ebe6bb0173cf6d0a603`.
The implementation does not call `wgpuInstanceWaitAny`: that symbol is an
unimplemented panic in this version, and the callback-producing functions
return null futures. Instead, local completion objects keep callback state alive
and `wgpuInstanceProcessEvents` polls the real instance devices. Callback
completion is published with release/acquire atomics, with the backend's existing
wait timeout preserved.

`cosmo_webgpu_initialize()` registers and selects the linked software Vulkan
provider before registry enumeration. Instance creation also calls this guard.
An unavailable embedded provider fails initialization; there is no attempt to
load an OS-native Vulkan library. Adapter requests explicitly select Vulkan and
a software adapter. No surface or display server is needed. The GGML device
retains the upstream GPU scheduling category so llama can offload to its distinct
WebGPU buffers; its description and the diagnostics API identify the actual
CPU/software adapter rather than implying physical GPU execution.

The original backend requires WebGPU `ShaderF16`, including for its matrix
kernels that stage F32 values in half-precision workgroup memory. This requirement
is checked before advertising a device. The adapter does not pretend that an
F32-only runtime satisfies it. Standard subgroups are requested only when
reported by the adapter. This pinned native C implementation does not map the
standard `Subgroups` feature or fill the standard subgroup-size information, so
that query returns false: the existing workgroup kernels are used, and
subgroup-dependent flash attention is rejected. No subgroup size is invented.
Dawn-only implicit synchronization, toggles and experimental subgroup matrices
are not requested. Packed integer-dot kernels
are disabled until this specific path is validated. Timestamp GPU profiling is
rejected at configure time because that optional API subset is not implemented.

Existing per-operation checks still determine which graphs the backend accepts.
The shared-runtime restrictions for the custom RoPE offset and SSM history
extensions remain in force. SD's F8 types and optimizer/backward extensions remain
in the shared GGML implementation; this adapter does not establish WebGPU
support for those operations or complete SD training. The existing WebGPU
`supports_op` rejects its unimplemented operations, allowing ordinary scheduler
fallback where the application permits it and preserving the trainer's explicit
no-fallback behavior.

`include/cosmo-webgpu.h` exposes process-lifetime diagnostic counters at real
backend graph, WebGPU dispatch, queue submission and mapped-readback call sites.
The matrix counter is a subset of dispatches using the original `mul_mat*`
pipeline labels. Counters alone do not prove correct computation: verification
must directly execute a graph on WebGPU, compare readback with an independent
reference, and inspect dispatch/submission deltas. Provider identity, physical
adapter kind and the foundation's actual native-loader-open count are separate
checks. The full application must additionally demonstrate model prefill and
decode using these backend calls.

The adapter and full patched GGML WebGPU translation unit have been compiled
with the Cosmopolitan 4.0.2 SDK and these exact WebGPU headers. Runtime shader,
numerical and same-artifact operating-system checks are recorded by the
application verification pipeline; compilation by itself does not establish them.
