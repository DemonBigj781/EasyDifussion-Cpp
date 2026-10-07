# All-in-one Easy Diffusion: scope and integration contract

The goal of `cosmopolitan-test` is one application executable containing Easy
Diffusion's native functionality, server and UI resources, with the same
application bytes runnable on supported Windows and Linux x86-64 machines.
Use C for application interfaces and new glue where practical; retain necessary
C++ or Rust implementations when rewriting them would delay or damage working
functionality.

This is a whole-application port. A tensor probe, language-model executable,
or bundled collection of subprocesses does not by itself complete it.

## Source baseline

- EasyDifussion-Cpp main: `b42704a624652a53099fbc994e9a6b7648cd3b0c`, including
  the native tools, plugin, diagnostic and training changes pushed on
  2026-10-06.
- llama.cpp: existing source snapshot
  `c589f0ed10c643678c4707dd160c21ac7633ebc0`.
- stable-diffusion.cpp: existing patched snapshot based on
  `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`.
- Its GGML base: `e20c3a14aa70ee84ca58499814206dd08d8026bc`, plus the
  repository's custom operations, optimization and training work.
- Theory documentation reference:
  `aa54da093aeeb9fa9ce34163f73d5797c31c316d`.
- Pinned Cosmopolitan software-Vulkan foundation, independently verified before
  this application integration:
  [cosmopolitan-lua/webgpu-cpu](https://github.com/DemonBigj781/cosmopolitan-lua/tree/c2a7b3f8c871e44ae1e3fc8db36759c7b1829326).

The two existing GGML snapshots are inputs to reconciliation. The final
application must compile one authoritative GGML implementation and give both
model engines and the trainer the same headers, tensor layout, operation
definitions, backend registry and allocator contract.

The [source update audit](docs/source-audit.md) compares those snapshots with
their actual upstream Git objects and distinguishes copied source from working
application integration. The [Colibri and KoboldCpp review](docs/resource-management.md)
records resource-sharing and application design references without importing a
second model runtime.

The [application services and API audit](docs/application-api.md) records the
existing route declarations, UI dependencies and remaining native service
work, including LibTorch/ONNX ownership and removal of deployed Python.

## Implemented WebGPU integration and provider selection

The build now links the existing WebGPU backend from the authoritative shared
GGML, a narrow C++ adapter over the native C API, wgpu-native, Mesa lavapipe
and LLVM into the application. Both model engines use this same GGML registry
and allocator contract. Original WGSL kernels remain responsible for WebGPU
computations; the adapter does not emulate them with the CPU tensor backend.
The application configures the native, embedded or automatic Vulkan provider
policy before backend discovery. Explicit embedded selection fails if that
provider is unavailable and does not load an OS-native Vulkan library.

[software-webgpu/PIN.json](software-webgpu/PIN.json) records the exact foundation
commit and archive hash: wgpu-native 29.0.1.1, Mesa 25.2.8, LLVM 19.1.7,
Cosmocc 4.0.2 and Rust `nightly-2026-07-28`. The build compiles the patched
Rust runtime and C compatibility objects, static driver and LLVM archives,
and the original C/C++ application sources. Python, Rust/C/C++ compilers,
CMake/Ninja, Mesa's generators and its host shader compiler are build-time
dependencies. Runtime shader compilation uses embedded LLVM. The recipe
requires this dependency set and cannot silently emit a CPU-only application
if it fails. The [build instructions](README.md#build) describe supported
staging, dependency-cache and compilation options.

The embedded path performs software compute on the CPU. It supplies a real WebGPU
command and shader implementation without requiring a GPU driver or display
server; it is not hardware acceleration. The ordinary GGML CPU backend is
also linked. `llama --backend cpu|webgpu` chooses the tensor backend, and
`sdkit --backend webgpu` uses the shared WebGPU device selected by the saved or
explicit `--provider` and `--device` settings. Native Vulkan discovery and
per-adapter selection are implemented for Windows and Linux. Automatic native
selection requires a compatible discrete/integrated GPU; explicitly selected
native software adapters remain labeled software. Physical GPU execution has
not been established by the available software-only validation. The routing
change is not proof of every diffusion graph's support. CUDA, Metal, SYCL and
other accelerator paths remain separate integration work.

The original WebGPU matrix kernels require `ShaderF16`, including when their
input tensors are F32. An adapter without that capability is not registered
as a usable WebGPU device. The pinned native C implementation does not expose
standard subgroup support or subgroup-size information, so the backend uses
its workgroup kernels and rejects subgroup-dependent flash attention.
Dawn-only experimental subgroup matrices and toggles are omitted; packed
integer-dot kernels and optional GPU timestamp profiling are disabled.
Existing operator, shape and type checks remain authoritative, including the
shared-GGML restrictions for custom RoPE offsets and SSM history. See
[backend/README.md](backend/README.md) for exact API and capability limits.

The [completed inference record](../COSMOPOLITAN_INFERENCE_VALIDATION.md) covers
[run 37551341258](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37551341258),
built from clean source `6898a4e564c3363d1692f0aed1a2da0395a17a77`. The same
executable passed shared-GGML WebGPU tensor/text gates, SD 1.5 image generation
on CPU and software WebGPU, and CPU native HTTP generation/progress/cancellation
on native Windows and Linux. Linux isolated/bootstrap checks remain separate
from the external-model image/API runs, which are not filesystem-isolated.
The few-step image outputs establish completed inference, not visual quality.
End-to-end training, other model families, WebGPU HTTP inference, application
parity and hardware acceleration remain separate gates. Earlier integration
and standalone foundation records retain their original scope.

## Full application inventory

| Component | Existing location | Port responsibility |
| --- | --- | --- |
| Shared tensor engine | Both vendored `ggml` directories | Reconcile APIs and custom operations; one implementation and symbol set |
| Language inference and tokenization | `source/llama.cpp` | Static model engine, C API, safe model loading, no external llama-server requirement |
| Diffusion/video inference | `source/sdkit3-port-source/stable-diffusion.cpp` | Preserve local model, conditioning, attention, caching and conversion changes |
| Native server/model routing | `source/sdkit3-port-source/src` | In-process server, model lifecycle, generation, interruption and device routing |
| C++ UI and static resources | `source/UI.cpp`, `ui/media`, UI plugins | Link renderer, embed assets, serve pages, preserve settings and plugin behavior |
| Application HTTP API | `ui/easydiffusion/server.py` and route modules | Replace Python handlers while preserving request/response and error behavior; document OpenAPI contracts |
| Queue and job lifecycle | `ui/easydiffusion/task_manager.py`, `tasks`, training service | Scheduling, progress, cancellation, cleanup, persistent history and resource contention |
| Partial native SD 1.5 LoRA training | `tools/sd15_lora_train.cpp` and custom GGML/model code | Same process and GGML; retain custom backward math, AdamW rates, export and sampling behavior |
| Python training | `training/trainer.py`, `training/service.py`, sd-scripts integration | Native replacements for remaining training families and options before removing the Python path |
| Native Sprite-GPT | `tools/sprite_gpt.cpp` | LibTorch model/autograd/optimizer dependencies and device/runtime portability |
| Native vision | `tools/native_vision.cpp` | LibTorch/TorchScript loading, preprocessing, decoding and NMS |
| WD14 tagging | `tools/wd14_tagger.cpp` and tagging routes | ONNX Runtime dependency or an explicitly validated replacement; retain captions/labels |
| Model conversion and Transformers | llama conversion modules, `model_tools.py`, TIPO | Replace required model/tokenizer/configuration/conversion behavior, not merely Python syntax |
| Gallery/files/configuration/plugins | Python modules and UI plugins | File discovery, saving, metadata, preferences and existing integration contracts |
| Online services | model browser, Perchance, other service modules | Preserve network/API behavior through native HTTP/JSON clients |
| Accelerator runtimes | GGML backends, Vulkan foundation, CUDA/other native APIs | Per-device capability detection and validated execution; keep CPU fallback |
| Specialized attention | Theory notes and current native attention code | Preserve semantics and training requirements; see [attention.md](docs/attention.md) |

## Partial SD 1.5 training is deliberately still partial

The repository's [training documentation](../training/README.md) is the
behavioral baseline. The native path includes custom UNet/CLIP LoRA handling,
per-parameter AdamW rates, forward sampling that must not update weights,
learning-rate schedules and safetensors export. It does not establish full
training parity with the Python implementation.

In particular, preserve the documented restrictions around native checkpoint
resume, AdamW8bit, selectable precision, additional optimizer settings and
training families. Do not substitute unsupported optimizer settings silently.
SDXL/Anima Python training, embeddings and the LibTorch Sprite-GPT path require
their own native work and validation.

Small backward/optimizer regression tests protect the shared GGML merge.
They do not prove a full training run, checkpoint quality, GPU sampling,
or compatibility with every prepared training recipe.

The portable trainer defaults explicitly to the CPU backend. Linking WebGPU
does not add implementations for all custom backward operations, F8 tensor
types or optimizer kernels. Its operation capability checks and explicit
no-fallback behavior must continue to reject unsupported training graphs.
The new WebGPU inference tests do not certify full SD 1.5 training, native
resume, selectable mixed precision, AdamW8bit or any additional training family.

## LibTorch, Transformers and OpenAPI

LibTorch is already used by native C++ components. It can remove a Python
frontend dependency for some workloads, but still carries ATen, tensor,
autograd, serialization, threading, dispatcher and optional device-library
requirements. The ordinary OS-specific LibTorch distributions are not proven
Cosmopolitan libraries. A direct port must account for that dependency closure;
a replacement must prove equivalent behavior for the selected models.

A Transformers replacement is a model- and feature-specific effort. Native
llama.cpp covers its supported architectures and tokenizers. It does not
automatically replace arbitrary Transformers pipelines, processors, conversion
scripts, multimodal models or training code.

OpenAPI is tracked as the application's HTTP schema and compatibility contract.
If another API was intended by that name, add it separately rather than conflating
it with OpenAI-compatible endpoints, OpenCL or OpenVINO.

## Runtime ownership

One process owns the shared backend registry. Each task has explicit model,
tensor-buffer and graph lifetimes. User-visible operations must not reset global
backends while another operation still holds buffers or contexts.

Native generation now has a server-owned coordinator with single-job admission.
It copies request state, retains the request gate and temporary options, and
posts completion back to the originating HTTP I/O thread. Model work runs on
an explicitly requested and measured 8 MiB pthread stack; the coordinator
joins it and transfers exceptions and allocation-error state before responding.
This leaves progress and task-specific interruption responsive. Cancellation
during model loading is not implemented. Shutdown closes admission and joins
owned work before server dependencies are destroyed, then abandons undelivered
responses without further I/O. These paths are described in
[application-api.md](docs/application-api.md).

A broader queue must still arbitrate inference, training, tagging and
conversion memory use.
Sharing a library does not imply that two unrelated model checkpoints use the
same weight allocation. Memory sharing is valid only when representation,
lifetime and mutability agree. Training must not overwrite inference weights
or caches accidentally.

C interfaces should carry explicit error results, cancellation/progress
callbacks and documented memory ownership. Exceptions remain internal to C++
adapters and must not cross a C call boundary.

## Packaging and completion criteria

The executable may contain resources such as HTML, JavaScript, CSS, fonts,
licenses and a small validation model. Large user models, datasets, outputs and
configuration remain data. A host browser may display the local UI.

Build-time Python/CMake tooling is distinct from a deployed Python runtime.
The final application must not require Python, PyTorch Python wheels,
externally compiled backend executables or the source checkout to perform its
supported native features. Windows invokes the PE directly. Linux's `/bin/sh`
entry point extracts the bundled APE loader temporarily using ordinary host
utilities; the separate isolated test uses an explicit APE loader. The
application's bytes remain unchanged in both launch modes. The embedded
software path has no external Mesa/Vulkan dependency. The implemented native
provider requires suitable system GPU drivers; on Linux the pinned SDK also
needs a host C compiler for its first-use native library helper. See
[configuration](CONFIGURATION.md) for platform commands and prerequisites.

The mandatory WebGPU completion gate is execution, not enumeration:

- Direct shared-GGML WebGPU graphs run F32 and Q4_0 matmul followed by bias,
  RMSNorm, SiLU and softmax for two input variants each. All twelve readbacks
  are compared with independent scalar references. The probe uses WebGPU
  buffers and direct backend computation, without a CPU scheduler fallback,
  and checks that an unsupported backward operation is rejected.
- The trained embedded llama fixture must tokenize, prefill and decode sixteen
  steps with WebGPU selected, finite logits and the independently established
  greedy-token sequence. Ordinary model scheduling may still use supported
  CPU operations; the inference check additionally requires actual WebGPU
  matrix dispatches during inference, not a claim of universal offload. A
  separate snapshot after synchronized prefill must show positive graph,
  submission, dispatch, matrix-dispatch and readback deltas for the sixteen
  decode steps themselves; prefill activity cannot satisfy that requirement.
- Each test requires real graph/submission/dispatch/matrix/readback counter
  deltas, an explicitly identified software adapter and zero native Vulkan
  library opens. Counters accompany numerical output and are not sufficient
  evidence by themselves.
- Six additional alias-placement cases make twelve independent scalar checks
  across scheduler-allocated storage, preallocated CPU storage and preallocated
  WebGPU storage. In-place operations must stay with their canonical storage
  owner on a compatible backend before inputs are copied between backends.
- `webgpu-test` runs this tensor/model pair. `--self-test` requires both in
  addition to the alias cases and existing CPU, shared-GGML, training-math
  and application regressions. Windows direct-PE and Linux isolated/bootstrap
  jobs must pass these checks using the same packaged executable hash.

The direct tensor inputs are exactly representable in f16 because the original
matrix kernels use half-precision staging. Their scalar checks establish the
tested operations' correctness, not arbitrary-F32 numerical equivalence to
the CPU backend or coverage of all operators and model architectures.

The completed image gate loads the pinned full SD 1.5 Q4_0 checkpoint and
independently decodes four-step CPU and two-step software WebGPU PNGs at
256×256 on both OSes. Generation and denoising-only counters separately prove
WebGPU dispatch/readback; diffusion CPU fallback is allowed and unmeasured.
The CPU HTTP gate generates a PNG, observes 32 responsive progress requests,
checks option/task ownership and cancels a separate eight-step request during
sampling. It does not validate WebGPU HTTP inference or a real browser session.

Completion requires all of the following:

- One application artifact is built and used unchanged by Windows and Linux
  validation jobs.
- Inference, supported training and the UI/API operate through that application.
- Shared GGML symbols and header/layout consistency are verified.
- Application API parity is tracked route by route, including error and
  cancellation behavior.
- Python/Transformers-dependent features are replaced and validated before their
  original implementation is removed.
- Native training limitations and every remaining LibTorch/ONNX/GPU dependency
  remain visible until resolved.
- Unsupported accelerators or attention forms report an explicit reason and
  use a correct supported fallback where one exists.

The completed text, image and CPU HTTP gates are evidence toward these criteria.
They must not be presented as whole-application completion.
