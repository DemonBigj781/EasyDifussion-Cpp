# Foundation-first integration work

This development branch starts at `cosmopolitan-test` commit
`0f10cd4ab32213fe90e69c14d48de353ef698577`. It isolates new work from
`main`, `Theory`, and `cosmopolitan-test`; it does not replace any of them.

The implementation authority remains the existing project documents:
[PORTING.md](../../PORTING.md), [application services](../application-api.md),
[shared GGML](../../shared-ggml/README.md), [attention](../attention.md),
[native HF/LoRA conversion](../../../docs/native-hf-lora-gguf-plan.md),
[training](../../../training/README.md), and the Theory branch's architecture
and audit documents. This file records sequencing and integration boundaries,
not a smaller replacement scope.

## Current state

- The isolated development branch has been created.
- The three branch trees and all 67 workflow definitions in the Cosmopolitan
  snapshot have been reviewed. See [source baselines](source-baselines.json).
- No application code, dependency pin, runtime, UI, training implementation,
  or workflow is changed by this initial checkpoint.
- No application/dependency build, model inference, software-WebGPU execution,
  physical-GPU test, paid resource operation, or workflow dispatch has been
  performed for this checkpoint.
- The newer Cosmopolitan foundation has not yet been integrated. The existing
  foundation pin remains intact until its compatible source/runtime/toolchain
  handoff is established.
- A [versioned static handoff validator](HANDOFF.md) is implemented. It binds
  declared compiler, header, CRT, C++ ABI and
  component identities to exported artifact bytes; it does not select a new
  toolchain or establish runtime correctness. Python bytecode compilation,
  whitespace checks and 91 non-model handoff/consumer/routing tests pass.
  These synthetic and mocked cases do not invoke a real SDK or model.
- The [explicit source-runtime build consumer](CONSUMER.md) is implemented
  behind a complete-export selection. It leaves the default SDK and pin
  unchanged and refuses incomplete/mixed inputs. No compatible export has
  been built or selected; foundation activation remains pending.

## Required order

1. Finish the Cosmopolitan foundation as a coherent source/runtime/toolchain
   unit, then integrate it into this Easy Diffusion development branch.
2. Account for and implement the capabilities represented across all three
   application branches. Preserve distinct behavior; do not substitute a
   wholesale branch merge for a feature and contract review.
3. Basic non-model tests, compilation and static inspection may occur during
   implementation. Do not launch CPU/GPU model inference. Broader runtime
   validation still needs its own deliberate scope and permission.
4. Run CPU validation before GPU validation. GPU behavior remains unverified
   until it is actually exercised under a separately authorized plan.

Paid GPUs, Vast.ai, and secret-enabled GPU runners are off-limits. Do not
dispatch existing runtime workflows during the implementation phase.

## Source preservation, not source-count parity

The Cosmopolitan snapshot contains nearly all historical main-source bytes:
615 added blobs, one removed workflow, and five modified blobs compared with
the pinned main snapshot. Its missing behavior therefore primarily requires
portable integration, native service implementation and validation, rather
than copying main wholesale.

Theory contains a different decomposition and changed frontend/service
contracts: compared with main it adds 622 blobs, removes 255, and modifies 231.
These are Git tree counts, not feature counts or proof of functional parity.
A file moved or present in a tree can still be incomplete or unreachable.

The source commits in this checkpoint are audit snapshots, not a freeze on
upstream development. Main may continue receiving minor edits. Re-read all
three upstream heads before integration, compare changes since these recorded
snapshots and reconcile them explicitly. Do not overwrite continuing work or
move any existing branch as a side effect of updating this development branch.
Reported but not yet submitted main changes include an Anima difference, UI
reorganization, and an additional LTX video model by Lightricks whose exact
version is not confirmed. These are pending inputs, not code assumed present in this snapshot.
Reconcile the actual diffs when available; do not guess a model version or
replace those changes with this branch's older files.
The reported LTX model run used an unspecified alternate execution path.
FP8 matrix multiplication was not tested and must remain unverified.

For every capability, retain the source implementation or an explicit
replacement mapping until native behavior and later validation are accounted
for. Track implementation and verification separately. A rendered page, a
declared route, successful compilation, or bundled Python converter does not
establish working end-to-end functionality.

Preservation areas include both UI experiences; image, video and text
inference; all model/conditioning/attention/cache variations; configuration
and plugins; galleries/files/metadata; job lifecycle and resource ownership;
model discovery/download/conversion/tokenizers; tagging/vision; and all
training families, data preparation, optimizers, sampling, export and resume.
The partial native SD 1.5 trainer is not assumed correct or complete.
Python-exclusive training and transformation paths need native implementations
or compatible custom interfaces before their deployed Python path is removed.

### Required hardware accounting

Preserve per-device backend choices, capabilities, compilation flags and memory
constraints. The currently identified targets are:

- NVIDIA V100; exact memory capacity and configuration remain unspecified.
- NVIDIA P102-100; exact memory capacity and configuration remain unspecified.
- Dell NVIDIA RTX 3090 24 GB, part 0N78PC, Ampere.
- Intel XG310 with four independent 8 GB devices. Do not treat this as one
  automatically pooled 32 GB device.
- Steam Deck `gfx1033`; preserve its integrated/shared-memory constraints.
- Radeon RX 580 2048SP; do not substitute the capability/compilation assumptions
  of a differently configured RX 580.
- Jetson AGX Xavier 32 GB developer kit: ARM64/Volta, JetPack 5/CUDA 11.4,
  shared system/device memory. Preserve existing Xavier build flags and its
  distinct toolchain/runtime requirements. An ARM64 manifest validator is not
  evidence of a completed ARM64 application port.

Direct Vulkan is an important backend alongside WebGPU, particularly for
older hardware. Preserve and account for both paths; Vulkan availability is
not equivalent to satisfying WebGPU's current ShaderF16/subgroup requirements.
These targets are requirements, not tested support claims. No GPU test or
paid resource use is authorized by this inventory.

FP8 compatibility is required on older devices, including the Ampere RTX 3090,
even when conversion or emulation is slower. Preserve the format's semantics
and distinguish model storage precision, conversion and actual arithmetic.
Do not infer Hopper-native FP8 capability for Ampere or report a fallback as
native FP8. Apply compatibility work only when the model/operation requires
it, avoiding conversion and overhead on paths that can run without it.

## Foundation handoff contract

The current application independently assumes Cosmocc 4.0.2 in several places.
Changing only `software-webgpu/PIN.json` would not establish a compatible port.

Review and update the following as one integration change:

- `build.py`: SDK source/archive hash, compiler-driver preprocessing wrapper,
  tool paths, cache identity, packaging tools and recorded build provenance.
- `software-webgpu/prepare.py`: imported source paths/archive identity,
  `seed_sdk`, Rust setup, generated linker, runtime object discovery and the
  WGPU build/output layout. Its existing `shim-*-x86_64.o` convention must not
  be silently reused if the foundation now supplies a source-built runtime.
- The C/C++ compiler, headers, CRT, libc/runtime archives, Rust standard library,
  compatibility objects, ABI and linker must identify the same compatible
  foundation. Do not combine newer runtime overrides with the old stock SDK.
- LLVM, Mesa, WGPU and native-provider loading/bridging must retain a complete
  static dependency closure for the embedded path, with licenses and hashes.
- `cmake/software-webgpu.cmake` consumes `LINK.json` fields
  `include_dirs`, `runtime_sources`, `runtime_objects`,
  `runtime_link_options`, `wgpu_archive`, and `driver_link_args`.
  Update this interface deliberately if the foundation manifest changes.
- Preserve one authoritative GGML, shared tensor layout, backend registry and
  allocator contract for llama, diffusion and training.
- Preserve both embedded software WebGPU and native-provider selection.
  Software WebGPU is CPU execution, not proof of physical GPU acceleration.
- Invalidate caches when source, compiler/runtime, ABI, flags, patches or
  dependency identity changes. Historical executable validation remains tied
  to its original artifact; it does not certify the new foundation.

The integration handoff must identify the exact source commit and all required
paths, compiler/runtime manifest, dependency manifest, runtime ownership/ABI
changes, reproducible source/archive identity, and unresolved implementation
gates. An unverified candidate branch is not a completed foundation.

## Workflow safety

In the pinned Cosmopolitan snapshot, the only push-triggered workflows are
`cosmopolitan-test.yml` and `cosmopolitan-vast-test.yml`. Both restrict their
push trigger to `cosmopolitan-test`, so pushes to
`dev/cosmopolitan-foundation-integration` do not activate them.
The remaining workflow definitions are manually dispatched or reusable.
This is a snapshot observation, not a guarantee about future workflow edits.

Do not extend those triggers to this branch or dispatch a workflow without
checking its complete steps and reusable dependencies. Even a build-labelled
workflow can run behavioral probes. The existing Vast approval input does not
override the prohibition on paid GPU resources.

## Completion evidence

Before a capability can be marked complete, record its source requirement,
native implementation and UI/API/CLI entry points, input/output and failure
contracts, resource/cancellation ownership, implementation status, and later
CPU/GPU validation results. Keep unrun stages explicitly unverified.
Historical numerical and image evidence must retain its original source,
artifact hash, platform, backend and limits.
