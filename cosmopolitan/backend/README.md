# GGML WebGPU through the static Cosmopolitan runtime

This adapter connects the existing shared GGML WebGPU backend to the pinned
wgpu-native C ABI, the embedded lavapipe/LLVM runtime, and installed Windows or
Linux Vulkan drivers. It preserves the
customized SD GGML as the authoritative implementation. The llama and SD
consumers use the same GGML registry, device, allocator and backend code.

`shared-ggml/patches/0004-wgpu-native-c-api-backend.patch` adapts that backend's
initialization and CMake dependency. It does not replace the WGSL kernels with
CPU tensor operations. The GGML C API remains `ggml_backend_webgpu_reg()` and
`ggml_backend_webgpu_init()`. The registry name is `WebGPU`. Patch
`0006-enumerate-webgpu-devices.patch` exposes each compatible adapter as a
separate `WebGPU0`, `WebGPU1`, etc. device with its own logical device, buffer
type and context. Those selectors describe the current registry order; they
are not persistent physical device identities.

`include/webgpu/webgpu_cpp.h` implements only the C++ API subset used by this
pinned GGML backend. Every resource and compute operation goes through the
actual wgpu-native C functions. Resources use shared ownership with one native
reference per owned handle. Descriptor conversion uses the exact pinned C
structures and initializer macros; it does not assume Dawn structure layout or
ABI compatibility. This is not a general implementation of Dawn's C++ API.

The pinned native adapter-list API has no output-capacity argument. Its count
and fill operations each enumerate independently, so a changing adapter list
can outgrow a caller's array. The application-owned patch in `patches/` adds
`cosmo_wgpu_instance_enumerate_adapters`, which writes at most the supplied
capacity and releases unreturned adapter handles. The wrapper releases partial
results and retries growth at most four times; it accepts a shrinking list.
Its five Rust unit cases exercise the same production ownership helper.
The build records the patch and generated Rust source hashes separately while
keeping the imported foundation archive pristine.

The foundation pin is owned by `../software-webgpu`. Its WebGPU API is
wgpu-native v29.0.1.1, revision `6aed50955d934ac36049ba8d002034841633ae02`,
with webgpu-headers revision `673658bc2bd70ec39fc55ebe6bb0173cf6d0a603`.
The implementation does not call `wgpuInstanceWaitAny`: that symbol is an
unimplemented panic in this version, and the callback-producing functions
return null futures. Instead, local completion objects keep callback state alive
and `wgpuInstanceProcessEvents` polls the real instance devices. Callback
completion is published with release/acquire atomics, with the backend's existing
wait timeout preserved.

`cosmo_webgpu_configure()` stores the provider/device policy before the registry
is initialized. `cosmo_webgpu_initialize()` only registers the linked software
driver. It does not open a host library. The instance factory serializes provider
selection and instance creation; existing instances retain their provider.
`auto` creates native and embedded instances together, enumerates both, and
prefers a compatible physical GPU. `embedded` restricts discovery to the
included software driver. `native` restricts discovery to the operating
system's Vulkan loader and requires a physical GPU for automatic selection.
An explicitly named native CPU adapter can be tested, but remains software.
See [configuration](../CONFIGURATION.md) for saved settings and CLI precedence.

The supported native loader paths are `vulkan-1.dll` on Windows and
`libvulkan.so.1` on Linux. The executable includes its inference libraries,
software driver and shader compiler; physical acceleration still depends on an
installed compatible graphics driver. On Linux, Cosmopolitan 4.0.2 also builds
a native `cosmo_dlopen` helper on first use and needs a host C compiler for that
step. Explicit embedded operation keeps the existing deployment without host
Vulkan libraries. No surface or display server is needed for these compute paths.

Discrete and integrated adapters use their actual GGML device categories.
Software adapters use a distinct accelerator buffer category and are labeled
`software` in HTTP device metadata; they are not advertised as physical GPUs.
Unknown adapter types cannot satisfy the hardware-required gate. WebGPU
allocation limits are not reported as free VRAM. Memory remains unknown when
the API cannot provide it. This pinned API exposes no stable UUID/LUID to this
wrapper, so `stable_id_available` is false and no synthetic ID is substituted.

Native Linux calls must run on the original main thread because a new
Cosmopolitan pthread does not receive host glibc TLS. Registry/device creation
is completed there before serving requests. The native main executor keeps the
whole model-load, inference and PNG closure there while Crow runs on a service
thread. Shutdown drains that work before destroying cached models on main.
SD's deferred checkpoint loader also runs its complete read/convert/upload task
on that original thread under Linux `auto` or `native` policy. Its former
`std::thread` workers could enter the native driver through a tensor upload,
even when configured for one worker. This serialization also applies to CPU
requests under those policies, so mixed native and embedded buffers never rely
on the last selected adapter. Explicit `embedded` policy and Windows retain
parallel loading. The existing host-only conversion/export caller retains its
workers through an explicit internal flag that rejects non-host destination
buffers before upload.
Native callbacks and validation/debug layers remain disabled because the
supported ABI bridge does not provide reverse callbacks into Cosmopolitan.
The auxiliary HTTP upscaling route returns 501 for Linux native/auto until it
uses the same executor. The CPU preprocessing route is unaffected.

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

The application's historical [software validation record](../../COSMOPOLITAN_WEBGPU_VALIDATION.md)
demonstrates direct scalar-checked GGML graphs and trained-model prefill/decoding
through this adapter on one unchanged Windows/Linux executable, built with the
Cosmopolitan 4.0.2 SDK and these exact WebGPU headers. The direct probe uses no
CPU scheduler fallback; ordinary llama scheduling still includes CPU buffers
and supported CPU operations. The recorded results apply to the tested software
adapter, fixture and operations, with the capability restrictions above.

The new `webgpu-device-test` command uses the same four F32/Q4_0 graphs and
twelve independent scalar comparisons with the configured device. Its
`--require-hardware` option rejects software and unknown adapters before
accepting execution. `--all` exercises each compatible device in one process,
including native and embedded provider coexistence. The separate native-driver
CI job uses an identified host software driver; passing it establishes native
loader, shader and inference execution on that driver, not physical GPU
validation. The production API also records counters scoped to each inference
closure in `NATIVE_INFERENCE_EXECUTION` alongside its actual execution thread.
