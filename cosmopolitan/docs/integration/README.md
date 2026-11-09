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
- No build, behavioral CPU test, software-WebGPU test, physical-GPU test, paid
  resource operation, or workflow dispatch has been performed for this checkpoint.
- The newer Cosmopolitan foundation has not yet been integrated. The existing
  foundation pin remains intact until its compatible source/runtime/toolchain
  handoff is established.

## Required order

1. Finish the Cosmopolitan foundation as a coherent source/runtime/toolchain
   unit, then integrate it into this Easy Diffusion development branch.
2. Account for and implement the capabilities represented across all three
   application branches. Preserve distinct behavior; do not substitute a
   wholesale branch merge for a feature and contract review.
3. Complete implementation before behavioral CPU validation. Compilation and
   static inspection may occur during implementation, provided they do not
   implicitly launch runtime probes or tests.
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
