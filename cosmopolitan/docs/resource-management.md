# Resource management: Colibri and KoboldCpp review

Reviewed on 2026-10-06 for the all-in-one Cosmopolitan port. This document
records design references; neither project's implementation has been imported
or demonstrated inside this executable.

## Pinned sources

- [Colibri](https://github.com/JustVugg/colibri/tree/ce370e87d7b623d7759b52ec2007d75fc5b0e87e),
  revision `ce370e87d7b623d7759b52ec2007d75fc5b0e87e`.
- [KoboldCpp](https://github.com/LostRuins/koboldcpp/tree/f9a32456edd3e5bc5fd6327fb1a370d647a2bff5),
  revision `f9a32456edd3e5bc5fd6327fb1a370d647a2bff5` on `concedo`.

The C project previously supplied for this idea was Colibri. "Kobold CCP" is
interpreted here as KoboldCpp.

## What each reference contributes

| Question | Colibri | KoboldCpp | Consequence for this port |
| --- | --- | --- | --- |
| Models larger than fast memory | Routes selected MoE experts through storage, RAM and optional GPU tiers | Offers offload, context, batch and memory-fitting controls | Separate model size from peak resident working set and transfer cost |
| One application with several AI features | Several model-family engines behind a common frontend | Text, diffusion and other media paths, several HTTP APIs and bundled UIs | Share application services and preserve each feature's request lifecycle |
| Mostly C | The inference engines and several interfaces are C | Model implementations and adapters include substantial C++ | New glue can use a C ABI without translating every working C++ kernel |
| Python removal | Launcher, API gateway and resource planner still include Python | The single-file package includes Python and native shared libraries | Reimplement required orchestration; neither package is the final deployment model |
| One identical Windows/Linux file | Ships platform-specific artifacts | Ships platform-specific artifacts | Retain Cosmopolitan's own build and validate the exact same artifact on both OSes |
| Shared GGML | A separate tensor/model implementation, not a GGML drop-in | Current code also retains legacy versioned GGML compatibility | Keep the explicitly reconciled GGML selected for Easy Diffusion |

Sources: both pinned READMEs; Colibri's [resource planner][coli-plan];
KoboldCpp's [packaging script][kcpp-package], [application interface][kcpp-api]
and [backend adapter][kcpp-backend].

## Colibri: transfer the resource policy

Colibri's principal useful idea is to distinguish *available model data* from
*weights resident in the fastest tier*. Selected MoE experts can be brought in
on demand; frequently used experts can stay resident; bounded prefetch can
overlap reads with computation. Its README describes per-layer caches, routing
history, batched expert requests and optional heterogeneous execution. Those
are project implementation claims, not performance measurements reproduced
for Easy Diffusion.

The [expert-store interface][coli-store] is particularly relevant to a C
application boundary. A lookup returns tensor views with an explicit lease;
release ends the lease; prefetch is advisory and cannot evict an actively used
slot. The interface makes concurrency guarantees the responsibility of each
implementation. This is a useful ownership pattern for our shared resource
manager, independent of Colibri's tensor representation.

Its [planner][coli-plan] reads model metadata to distinguish dense data,
routed experts and retained representations. The amount stored on disk need
not equal the allocation required after loading or conversion. A GGML-based
planner must make the same distinction using the actual selected model and
backend rather than estimating every allocation from the checkpoint's byte
count.

This is not a generic way to make arbitrary models cheap. A dense diffusion
UNet does not acquire MoE routing merely by using an expert cache. Repeated
weight streaming can trade capacity for substantial I/O latency. Training also
needs gradients, optimizer state and saved or recomputed activations; the
inference working set is not a training memory estimate.

### Portability and GPU limits

Colibri's [compatibility header][coli-compat] includes OS-specific mechanisms
for positioned reads, memory advice, locking and allocation. Those branches
target conventional platform builds. A Cosmopolitan adaptation should use its
portable interfaces where they supply the needed semantics, and test large
offsets, short reads and cancellation on both systems. Defining `_WIN32` for
the entire portable build would select a native Windows ABI rather than solve
the runtime portability problem.

The pinned [Vulkan documentation][coli-vulkan] requires a Vulkan 1.2 driver,
specific shader capabilities and build-time shader compilation. It describes
resident experts, dense projections and an attention path, while documenting
fallbacks and decode-focused limitations. Vulkan's name alone is insufficient
to establish model or device coverage. Our embedded software Vulkan provider
would need the same operation and capability checks before using these ideas.

The [GPU backend documentation][coli-gpu] distinguishes directly linked Linux
CUDA/HIP builds from a separate Windows DLL interface. It also distinguishes
device discovery from actual tensor residency and computation. Those are
useful validation lessons, but they do not prove an APE can call every vendor
driver. Each native driver path still needs an ABI adapter and device tests.

Colibri's optional cluster design keeps coordination local and sends grouped
expert work to workers. It is a later distribution reference, not a
requirement for the first single-machine application. Cross-machine work needs
explicit tensor serialization and failure recovery; process pointers cannot
be shared over that interface.

## KoboldCpp: transfer the application behavior

KoboldCpp is a close reference for packaging a local model application with
its UI and several API conventions. Its [native interface][kcpp-api] exposes
model loading, generation parameters, offload choices, context/cache controls,
draft models and diffusion settings. These are useful items to compare against
Easy Diffusion's feature inventory. The native header itself uses C++ features;
it is not a ready-made pure-C interface.

The [Linux packaging script][kcpp-package] uses PyInstaller's one-file mode
and includes Python application code, frontend resources and multiple `.so`
variants. This removes an installation step for users, but the implementation
still contains a Python runtime and OS-specific native components. Our
deployment goal requires those application responsibilities to run natively
inside the Cosmopolitan executable.

The [backend adapter][kcpp-backend] supports device selection and retains
`ggml_v2`/`ggml_v3` compatibility helpers. One part explicitly assumes one GPU
backend type when interpreting compile-time defaults. We should learn from
its public behavior without adopting that assumption or legacy symbol
duplication. This project needs one authoritative current GGML, while any
future accelerator registry must describe each concrete device independently.

The supplied [license notice][kcpp-license] separates the MIT upstream portions
from KoboldCpp-specific Python/C++ code and KoboldAI Lite under AGPLv3. This
review does not copy those components or infer a license for a combined work.
Any later code import needs its own provenance and retained notices.

## Proposed shared application contract

The following is a design target, not a claim that all consumers are already
connected to one scheduler.

| Resource | Owner | Sharing rule |
| --- | --- | --- |
| Backend registry and device inventory | Process runtime | Initialize once; do not tear down while any task holds a backend or buffer |
| CPU thread budget | Application scheduler | Account for llama, diffusion, training, I/O and UI responsiveness together |
| Model files and immutable weights | Model cache | Reuse only when content identity, tensor layout, quantization and backend representation match |
| Expert or staged-weight cache | Residency manager | Lease active entries; evict only after all dependent work completes |
| KV cache and recurrent state | Inference session | Reuse only for a compatible model, adapter, positional state and validated token prefix |
| Training parameters and optimizer state | Training task | Treat as mutable; inference sharing needs an explicit immutable snapshot or synchronization |
| Graph workspace and staging buffers | Active task or buffer pool | Return to the pool only after CPU work and device events are complete |
| LibTorch and ONNX allocations | Their adapters until integrated | Include in the global budget; do not assume their allocators or tensor layouts equal GGML's |
| UI/API progress and cancellation | Job manager | One job identity across loading, generation, training, saving and cleanup |

"Share everything" means a common implementation and coordinated resources.
It does not make unrelated model weights interchangeable. The C boundary
should make ownership, errors, tensor layout and completion explicit while
allowing the underlying C++ implementation to remain private.

For multiple GPUs, first distinguish independent jobs on different devices
from splitting one model across devices. Each needs its own memory accounting.
Splitting a model additionally needs supported transfers, synchronization and
operation placement. Several detected devices are not evidence that a single
generation uses all of them.

The direct GGML CPU backend is the initial compute path. Software Vulkan is a
separate compatibility path executed on the CPU; it does not create physical
VRAM or establish a speed advantage over native tensor kernels.

## Evidence required before enabling a new policy

Measure a representative workload with the model, source revision, device,
driver, prompt/data and settings recorded. Compare correctness first, then
peak RAM/VRAM, allocation failures, bytes transferred, time to first output,
steady throughput, cancellation and cleanup. Use both cold and warm caches
when evaluating streaming or reuse. A cache should improve placement without
silently changing the model's selected experts, quantization or training
updates.

The next implementation gate is the shared GGML build and same-file OS tests.
Expert streaming, automatic placement, multi-device execution and the full
application scheduler remain explicit follow-on work until they have their
own implementation and runtime evidence.

[coli-store]: https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/c/expert_store.h
[coli-plan]: https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/c/resource_plan.py
[coli-compat]: https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/c/compat.h
[coli-vulkan]: https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/docs/vulkan.md
[coli-gpu]: https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/GPU_BACKENDS.md
[kcpp-api]: https://github.com/LostRuins/koboldcpp/blob/f9a32456edd3e5bc5fd6327fb1a370d647a2bff5/expose.h
[kcpp-backend]: https://github.com/LostRuins/koboldcpp/blob/f9a32456edd3e5bc5fd6327fb1a370d647a2bff5/kcpp_backend.cpp
[kcpp-package]: https://github.com/LostRuins/koboldcpp/blob/f9a32456edd3e5bc5fd6327fb1a370d647a2bff5/make_pyinstaller.sh
[kcpp-license]: https://github.com/LostRuins/koboldcpp/blob/f9a32456edd3e5bc5fd6327fb1a370d647a2bff5/MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md
