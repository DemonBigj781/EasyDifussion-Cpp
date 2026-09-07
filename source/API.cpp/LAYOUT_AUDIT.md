# API.cpp layout and planning audit

This audit is the status map for `source/API.cpp`. It prevents an empty scaffold,
an old compatibility path, or a design document from being mistaken for the
current implementation. It records repository state as audited on 2026-09-06.

## How to resolve conflicting signals

Use these sources for different questions:

1. `LAYOUT.md` and this audit define intended ownership and placement.
2. Explicit CMake/Make source lists define what is compiled.
3. Runtime tests and `IMPLEMENTATION_STATUS.md` define what has actually been validated.
4. Feature README/design documents define contracts and known limitations.
5. Files under `docs/todo/` and files whose status says plan/design describe
   future work; they never prove that source exists.

If the build graph conflicts with the layout contract, record the path as
routing debt here. Do not silently redefine the architecture around the
accidental dependency.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| Canonical active | Current owning source path selected by a build or test. |
| Canonical partial | Correct owning path, but only part of its intended route exists. |
| Compatibility | Live adapter retained for an external or vendored interface; not the normalized API. |
| Prototype/reference | Research or superseded code excluded from production builds. |
| Deferred scaffold | Directory ownership is reserved, but no contract, build route, or support is claimed. |
| Superseded | An old path replaced by the listed canonical path. |
| Generated | Build output; never a source-of-truth path. |

## Current source map

| Area | Status | Authoritative interpretation |
| --- | --- | --- |
| `common/` | Canonical active | Library-wide backend identifiers and registries, not feature-specific behavior. |
| `features/detect/` | Canonical active | All Library handlers call feature Common. CPU, CUDA, OpenCL, OpenGL, Vulkan, and Mesa have local runtime proof; other backends remain source-audited only. |
| `features/load/` and `features/unload/` | Canonical partial | Common and `model/` CPU/CUDA byte-lifecycle routes are active. Other component directories are deferred placeholders, not parsers or graph loaders. |
| `features/overflow/` | Canonical active | Common owns planning and registered allocation/release dispatch. CPU/CUDA allocation behavior is runtime-tested; other backends remain unimplemented. |
| `features/attention/xformers/` | Canonical active for CPU/CUDA | CPU staged and CUDA fused translations share one Common contract. The stable-diffusion.cpp application adapter now reaches that same Common route. |
| `features/attention/flash/` | Canonical active for CPU/CUDA | CPU maps Common to the GGML native boundary. CUDA has a fused Common translation, a stable-diffusion.cpp application adapter, and local end-to-end proof; optimized GGML `fattn` remains a fallback compatibility path. |
| `features/attention/sage/` | Canonical partial for CUDA | Common owns the normalized `support` request and registered dispatch. The stable-diffusion.cpp GGML selector reaches that route, but forward still launches the direct GGML-native SM80 compatibility kernel. |
| `features/attention/{flex,split}/` and unimplemented backend branches | Canonical partial or deferred | Their design/location is reserved; support must be established from build and runtime evidence per backend. |
| `features/cache/`, `features/convert/`, `features/fuse/`, `features/gpu-zram/`, `features/memory/`, `features/ram/`, `features/read/`, `features/split/`, `features/swap/`, `features/vram/`, `features/write/`, `features/zram/` | Deferred scaffold | Empty or `.gitkeep`-only ownership markers. They are not active features. Top-level `split/` remains distinct from the Split Attention algorithm. |
| `features/ggml/`, `features/gguf/`, `features/reserve/` | Deferred scaffold | Explicitly reserved for later development. They have no build references and must not be pulled into current attention work. |
| `library/` | Canonical active | Public export/ABI packaging around Common. It contains no backend implementation tree and may call only Common. |
| `IMPLEMENTATION_STATUS.md` | Canonical active | Authoritative implementation and validation ledger. |
| `TODO_GRID.md` | Superseded compatibility pointer | Retained as a legacy planning grid; do not use it for current support claims. |
| `source/API.test/build/` and other build trees | Generated | Ignore when searching for canonical sources or build references. |

## xFormers ownership map

| Path | Status | Role |
| --- | --- | --- |
| `features/attention/xformers/common/` | Canonical active | Backend-neutral request, validation, capabilities, registry, staged calls, and fused-forward dispatch. |
| `features/attention/xformers/cpu/translation/cpu/` | Canonical active | Registered staged F32 Common implementation. No separate CPU definition layer is implemented yet. |
| `features/attention/xformers/cuda/definition/gpu/` | Canonical active | CUDA-native capability and request-validation contract. |
| `features/attention/xformers/cuda/translation/gpu/forward.cu` | Canonical active | The only CUDA translation registered with Common; fused F32/F16/BF16 input execution with F32 output. |
| `source/API.bridge/sdkit3-ggml/ggml-cuda/xformers-attention.{cu,cuh}` | Application adapter | Converts GGML tensors into the Common request and calls only Common; stable-diffusion.cpp's CUDA build compiles this Theory-owned source. |
| `features/attention/xformers/prototype/legacy-ggml-adapter.{cu,cuh}` | Prototype/reference | Superseded direct GGML/backend adapter, excluded from production builds. |
| `features/attention/xformers/prototype/cuda-*.cu` | Prototype/reference | Superseded materialized-stage CUDA implementation moved out of the production translation directory. |
| `features/attention/xformers/prototype/` | Prototype/reference | Excluded reference implementation. |
| `features/attention/xformers/native-calls/` | Prototype/reference | Research inventory, not source API or backend support. |
| `features/attention/xformers/{rocm,oneapi}/` | Deferred | Definition notes/placeholders exist; no registered translations or runtime claim exists. |
| `source/API.test/MultiTest/xformers/cycle/` | Canonical active test | Common contract and model byte-lifecycle cycle tests. |
| `source/API.test/MultiTest/xformers/generate-image/` | Canonical active integration test | Real stable-diffusion.cpp generation through the GGML application adapter and the same Common CUDA route. |
| `source/API.test/Feature/Attention/Xformers/Cpu/Main.cpp` | Superseded | Replaced by `MultiTest/xformers/cycle/xformers-load-unload-cpu-test.cpp`. |

There is one production CUDA xFormers backend route. Standalone tests and the
stable-diffusion.cpp adapter both enter the same Common dispatch and registered
CUDA translation. The successful generator therefore proves application
integration as well as the Common route.

## FlashAttention ownership map

| Path | Status | Role |
| --- | --- | --- |
| `features/attention/flash/common/` | Canonical active | Backend-neutral tensors, parameters, execution context, capabilities, validation, registry, and forward dispatch. |
| `features/attention/flash/cpu/` | Canonical active | Registered Common translation and GGML CPU compatibility definition; build/native-boundary smoke evidence only. |
| `features/attention/flash/cuda/definition/gpu/` | Canonical active | CUDA-native capability, shape, dtype, stride, and launch-bound contract. |
| `features/attention/flash/cuda/translation/gpu/forward.cu` | Canonical active | Registered fused online-softmax Common translation for F32/F16/BF16 input and F32 output. |
| `source/API.bridge/sdkit3-ggml/ggml-cuda/flash-attention.{cu,cuh}` | Application adapter | Converts supported GGML FlashAttention operations into Common requests and observes completed Common launches without calling a backend layer; stable-diffusion.cpp's CUDA build compiles this Theory-owned source. |
| `features/attention/flash/cuda/translation/gpu/fattn*.{cu,cuh}` and `template-instances/` | Compatibility | Retained optimized GGML fallback for unselected or Common-unsupported operations; it does not establish Common routing. |
| `source/API.test/Feature/Attention/Flash/Cuda/Main.cu` | Canonical active test | Common-only numerical and validation test with RTX 3060 runtime and Compute Sanitizer evidence. |
| `source/API.test/MultiTest/flash/cycle/` | Canonical active test | CUDA model-byte and Flash-result API roundtrip test covering 32 Common load/forward/unload cycles. |
| `source/API.test/MultiTest/flash/generate-image/` | Canonical active integration test | Real stable-diffusion.cpp generation through the GGML Flash adapter and normalized Common CUDA route. |

The normalized CUDA `forward.cu` route and GGML `fattn` compatibility fallback
remain deliberately distinct. The application adapter selects Common with
`SD_CUDA_FLASH_COMMON=1`; a 512x512 one-step generator recorded 40 Common
launches. The counter makes silent fallback a test failure.

## Test-tree audit

The CPU/CUDA xFormers cycle sources and CPU/CUDA generator sources are active.
Detect has active backend-specific tests for CPU, CUDA, ROCm, oneAPI, OpenCL,
OpenVINO, OpenGL, Vulkan, Mesa, and DirectML; compiler/runtime evidence still
varies by `IMPLEMENTATION_STATUS.md` status. The CUDA FlashAttention feature,
roundtrip, and image-generation tests are active; the image test shares the
stable-diffusion.cpp generator harness with xFormers.

The empty `Load/{mesa,opengl,vulkan}`, `Unload/{mesa,opengl,vulkan}`, and
`Overflow/{mesa,opengl}` directories are placement reservations only.
`MultiTest/{flex,sage,split}` directories remain unwired. The xFormers
Mesa/OpenGL/Vulkan placeholder sources are also
not wired tests. The xFormers cycle `CMakeLists.txt`, shared header, and the exact
`xformers-load-unload-test .cpp` filename remain unwired scaffolding; do not
rename or interpret them as support without confirming the intended contract.

## Deferred GGML, GGUF, and reserve boundaries

- `features/ggml/` is a future normalization candidate. It does not replace the
  vendored GGML trees, the current stable-diffusion compatibility adapters, or
  the shared-GGML ABI work in `docs/todo/32-merge-llama-stable-diffusion.md`.
- `features/gguf/` is a future normalization candidate. It does not implement
  GGUF parsing, writing, or conversion. The existing native converter plan
  currently proposes code under `source/sdkit3-port-source/src/conversion/`;
  ownership must be reconciled before this scaffold is promoted.
- `features/reserve/` is reserved for a future normalized resource-reservation
  contract. It does not currently reserve RAM, VRAM, devices, threads, or disk.
  It must not take over Overflow's fallback selection or migration policy.

Promotion requires a scoped contract, named methods, build wiring, tests, and a
status update in this audit and `IMPLEMENTATION_STATUS.md`. Adding a source file
or removing a `.gitkeep` alone is insufficient.

## Memory-plan ownership

| Concern | Owner/status |
| --- | --- |
| Cross-tier pressure, fallback order, allocation/release coordination | `features/overflow/`; CPU/CUDA partial implementation. |
| Capacity, accounting, and residency views | Future `memory/`, `ram/`, and `vram/` contracts. |
| Secondary-GPU storage mechanism | Future `gpu-zram/`; the CUDA secondary-VRAM path currently lives under Overflow until that interface exists. |
| Host compressed memory | Future `zram/`. |
| Disk-backed final tier | Future `swap/`. |
| Safety margins/resource reservation | Future `reserve/`; requirements appear in the memory plans but no API contract exists. |
| Project requirements and acceptance criteria | `docs/memory-oversubscription.md`, `docs/legacy-gpu-vram-memory-tier.md`, and TODOs 33/34; plans, not compiled ownership. |

Overflow supersedes the idea that an individual tier should choose the global
fallback order. It does not make the future tier/accounting directories
obsolete.

## Superseded and compatibility paths

- `features/attention/split_attention/` -> `features/attention/split/`.
- old API.cpp root backend directories -> feature/backend layers. The unused
  `library/backends/` notes/compatibility tree was removed.
- old API.cpp root `include/api` and `src` -> `library/include/api` and
  `library/src`.
- old `API.cpp/cuda/attention/*` ownership -> each attention feature's
  `cuda/translation/gpu/` directory.
- `source/API.bridge/sdkit3-ggml/ggml-cuda/{xformers,flash}-attention.*` are
  live application-to-Common adapters compiled by stable-diffusion.cpp's CUDA
  build. Flash retains the optimized GGML `fattn` fallback for operations not
  selected for or supported by Common. Sage selection uses Common `support`,
  while its direct GGML-native forward launch remains routing debt until the
  Common migration is complete.

Do not remove a compatibility shim until every external include/build reference
has migrated. Do not add new implementation to a shim.

## Checklist for later sessions

1. Read `README.md`, `LAYOUT.md`, this audit, `IMPLEMENTATION_STATUS.md`, and
   `COMMON_ROUTING_AUDIT.md` before moving or implementing a feature.
2. Search source build files while excluding generated build directories.
3. Classify the path using the status vocabulary above.
4. Preserve the single application -> Common -> translation -> definition route.
5. Treat `.gitkeep`-only branches as unsupported future ownership.
6. Record routing exceptions instead of copying their pattern.
7. Update this audit whenever a scaffold is promoted, a path is superseded, or
   the build graph changes ownership.
