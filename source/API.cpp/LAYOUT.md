# API.cpp architecture

## Purpose

`source/API.cpp` is the backend-normalization and Common library boundary
for Easy Diffusion. Backend-native APIs are described by definitions, adapted
by translations, and exposed only through feature Common code.

```text
Easy Diffusion <-> feature Common exported API <-> backend translation <-> backend definition <-> native runtime/device
```

Results return through the reverse path. A definition-to-translation-to-Common
diagram elsewhere describes ownership/normalization, not runtime call direction.

For CUDA, the concrete flow is:

```text
Easy Diffusion <-> Common <-> cuda/translation/gpu <-> cuda/definition/gpu <-> CUDA
```

## Root layout

```text
source/API.cpp/
├── Makefile
├── cmake/                         # root API.cpp build helpers
├── common/                        # library-wide backend registry/contracts
├── features/                      # feature-specific normalization
│   ├── attention/
│   ├── cache/
│   ├── convert/
│   │   ├── civitai/
│   │   ├── hf/
│   │   ├── lora/
│   │   └── model/
│   ├── detect/
│   ├── fuse/
│   ├── ggml/                      # future-only scaffold
│   ├── gguf/                      # future-only scaffold
│   ├── gpu-zram/
│   ├── load/
│   │   ├── clip/
│   │   ├── clip-vision/
│   │   ├── condition/
│   │   ├── image/
│   │   ├── latent/
│   │   ├── mask/
│   │   ├── model/
│   │   ├── unet/
│   │   └── vae/
│   ├── memory/
│   ├── overflow/
│   ├── ram/
│   ├── read/
│   ├── reserve/                   # future-only scaffold
│   ├── split/
│   ├── swap/
│   ├── unload/
│   │   ├── clip/
│   │   ├── clip-vision/
│   │   ├── condition/
│   │   ├── image/
│   │   ├── latent/
│   │   ├── mask/
│   │   ├── model/
│   │   ├── unet/
│   │   └── vae/
│   ├── vram/
│   ├── write/
│   └── zram/
└── library/                       # packaging for the exported Common surface
    ├── include/api/
    └── src/
```

`cmake/` and `Makefile` stay directly under `source/API.cpp`. Backend implementation directories, public include trees, and Library sources must not be reintroduced at the API.cpp root.

## Canonical feature layout

Every top-level feature family must use the same definition, translation, and
Common route. Families with concrete subtypes, such as attention,
cache, convert, load, and unload, apply the route to each subtype. A family
without subtypes applies it directly at the family root.

```text
features/[family]/[feature-type]/
├── README.md                      # optional status/usage notes
├── DESIGN.md                      # optional normalization contract
├── common/
│   └── [method].cpp               # one backend-common behavior
├── cpu/
│   ├── definition/
│   │   └── cpu/
│   │       └── [method].cpp
│   └── translation/
│       └── cpu/
│           └── [method].cpp
├── cuda/
│   ├── definition/
│   │   ├── cpu/
│   │   ├── gpu/
│   │   └── npu/
│   └── translation/
│       ├── cpu/
│       ├── gpu/
│       └── npu/
├── oneapi/
│   ├── definition/
│   │   ├── cpu/
│   │   ├── gpu/
│   │   └── npu/
│   └── translation/
│       ├── cpu/
│       ├── gpu/
│       └── npu/
├── rocm/
│   ├── definition/
│   │   ├── cpu/
│   │   ├── gpu/
│   │   └── npu/
│   └── translation/
│       ├── cpu/
│       ├── gpu/
│       └── npu/
└── prototype/                     # optional, non-production reference code
```

Each `[device-type]` leaf contains method files such as `[method].cpp`; device types currently include `cpu`, `gpu`, and `npu`.

Recognized backend families are `cpu`, `cuda`, `rocm`, `oneapi`, `opencl`,
`openvino`, `opengl`, `vulkan`, `mesa`, and `directml`. Every family follows the
same definition/translation pattern, but it creates only device-type branches
that its native API can represent honestly.

## Top-level operation families

The current `features/` directory set is `attention`, `cache`, `convert`,
`detect`, `fuse`, `ggml`, `gguf`, `gpu-zram`, `load`, `memory`, `overflow`,
`ram`, `read`, `reserve`, `split`, `swap`, `unload`, `vram`, `write`, and
`zram`.

`convert/` is divided by conversion target or source domain. `load/` and
`unload/` are divided by the component whose lifetime they control. The
remaining operation families are reserved normalization boundaries until their
method contracts are audited. An empty directory or `.gitkeep` records intended
ownership only; it does not advertise implementation or backend support.
`ggml`, `gguf`, and `reserve` are explicitly future-only scaffolds and must
remain outside current feature work until their contracts are approved.

`memory/` represents unified-memory hardware such as an APU, where CPU RAM and
GPU-visible memory are views of the same physical pool. Its accounting must not
add the RAM and VRAM views together or migrate data between aliases of the same
allocation. `ram/` and `vram/` retain the access, capacity, allocation, and
residency contracts for their respective views. On a discrete-GPU system they
remain separate physical tiers rather than being falsely reported as one APU
pool.

`overflow/` owns the policy for spilling remaining resources when the active
memory model—an APU shared pool or discrete RAM/VRAM tiers—has no room. Overflow
coordinates those contracts and the fallback storage mechanisms rather than
embedding one backend's allocator directly.

The declared Overflow fallback order is:

1. unused VRAM on another available GPU;
2. another GPU exposed through the `gpu-zram` compressed/swap-like tier;
3. available system RAM;
4. available zram, when it exists;
5. disk-backed swap as the final tier.

The normalized `plan`, `allocate`, and `release` methods now implement the CPU
heap route and the CUDA active-VRAM, secondary-CUDA-VRAM, Managed Memory, and
mapped-host routes. Candidates are backend-qualified so future ROCm, oneAPI,
OpenCL, and Vulkan translations can contribute mixed-vendor secondary devices
without changing the common policy. `gpu-zram`, zram, swap, and non-CUDA
secondary allocation remain unimplemented until their owning translations and
runtime tests exist.

`gpu-zram/`, `swap/`, and `zram/` own their respective storage mechanisms.
`gpu-zram/` is backed by secondary-GPU memory; `zram/` is the optional host
compressed-RAM tier; `swap/` is the disk-backed final tier. They expose
capabilities to Overflow but do not own the cross-tier selection order.

Top-level `features/split/` is a general operation family. It is distinct from
`features/attention/split/`, which owns the Split Attention algorithm and its
`plan`, `qkt`, online-softmax, and merge semantics. Neither directory may absorb
the other's contract merely because both use the word "split."

`features/detect/` owns compute-backend and device discovery. Its normalized
method is `detect`: backend availability, runtime-visible device enumeration,
identity, device class, architecture or compute version, and a point-in-time
memory snapshot. Root `common/` remains reserved for library-wide registry and
dispatch contracts. Model-format inspection, ControlNet preprocessing, and
image-object detection are outside this feature.


## Layer responsibilities

### Definition

`[backend]/definition/[device-type]/[method].cpp` records the backend API's native meaning, supported shapes/dtypes, capability constraints, synchronization, fallbacks, and equivalent procedure. A definition does not become the application API.

### Translation

`[backend]/translation/[device-type]/[method].cpp` adapts that backend/device definition to the feature's normalized common contract. Backend SDK calls, kernels, layouts, workarounds, and compatibility bridges stay here.

Production CUDA attention sources therefore live in:

- `features/attention/flash/cuda/translation/gpu/`
- `features/attention/sage/cuda/translation/gpu/`
- `features/attention/xformers/cuda/translation/gpu/`

### Common

`common/[method].cpp` is the all-in-one library middleman. It is the only
feature layer allowed to communicate with the application. Common owns
normalized validation, capability selection, backend dispatch, fallback policy,
shared sequencing, error normalization, and results. It must not expose CUDA,
HIP, SYCL, Vulkan, CPU-runtime, or NPU-native objects to callers.

`features/attention/common/` contains contracts shared by multiple attention types.

### Export packaging

`library/include/api/` and `library/src/` may package compatibility
or ABI-facing declarations for the exported Common library. They are not a
separate routing layer and may call only Common. Backend definitions and
translations may never include or call them.

## Current feature families

Attention types are `flash`, `flex`, `sage`, `split`, and `xformers`. A feature
implements only the methods it needs. Their normalized method inventories are:

- xFormers: `qkt`, `mask`, `softmax`, `av`, and `forward`;
- Sage: `support`, `key_mean`, `quantize`, `qkt`, `mask`, `softmax`, `sv`,
  `normalize`, and `forward`;
- Flash: `support`, `workspace`, `select`, `qkt`, `mask`, `softmax`, `av`,
  `combine`, and `forward`;
- Flex: `mask_size`, `validate`, `sum_heads`, `margin`, `normalize`,
  `threshold`, `pool`, `expand`, and `select`;
- Split: `plan`, `split`, `qkt`, `mask`, `softmax`, `av`, `merge`, and `forward`.

The planned cache types are EasyCache and TeaCache. `features/cache/` is
currently only a scaffold; the following method inventories are design targets,
not current source or support claims:

- EasyCache: `support`, `validate`, `reset`, `init`, `enabled`, `sigma`,
  `begin_step`, `active`, `skipped`, `has_cache`, `store`, `apply`, `before`,
  and `after`;
- TeaCache: `support`, `configure`, `validate`, `init`, `enabled`, `begin_step`,
  `rel_l1`, `rescale`, `accumulate`, `reuse`, `store`, `apply`, `skipped`,
  `before`, and `after`.

Compute-backend detection uses the single normalized `detect` method. Native
routes exist for CPU, CUDA, ROCm, oneAPI, OpenCL, OpenVINO, OpenGL, Vulkan,
Mesa, and DirectML. Native probing lives in each definition, translations map
results into the Common contract, and Common enforces and combines result
invariants behind one `detect(Backend)` entry point. OpenGL and Mesa are
explicitly context-bound. Missing portable free-memory queries remain unknown
rather than being inferred, and DirectML does not count shared system memory as
dedicated VRAM.

Model lifecycle currently uses the load and unload methods. The validated CPU
route copies model bytes into owned host storage; the validated CUDA route
copies them into owned device storage. Both return the same Common resource
contract, and unload clears that resource only after its owning backend
releases it. This byte-lifecycle contract does not claim model parsing, tensor
construction, or support for the still-empty clip, clip-vision, condition,
image, latent, mask, UNet, or VAE component routes.

These filenames describe normalized semantic methods, not mandatory separate
kernel launches or a claim that an empty scaffold is implemented. Each cache
type must reproduce the same definition, translation, and Common-facing flow.

See `IMPLEMENTATION_STATUS.md` for the backend/feature coverage matrix and
`LAYOUT_AUDIT.md` for the audited active/future/compatibility map. A matrix cell
is not supported merely because its empty directory scaffold exists.

## Placement rules

1. Keep feature and method names lowercase.
2. Put native backend semantics in `definition/[device-type]`.
3. Put backend-to-common adapters and kernels in `translation/[device-type]`.
4. Put only normalized cross-backend behavior in feature `common/`.
5. Do not place method source files directly in a backend, `definition`, or `translation` directory.
6. Keep prototype code out of production CMake targets.
7. A file is not runtime-supported until an explicit build target selects it and tests validate its backend/device.
8. Do not pre-create a complete backend/device matrix for a future feature.
   Keep its root README until a contract establishes the branches it needs.
9. Packaging under `library/` is Common-only and contains no backend
   implementation or compatibility backend tree.

## Migration notes

- `features/attention/split_attention/` became `features/attention/split/`.
- Root `cpu`, `cuda`, `oneapi`, `rocm`, and `vulkan`
  directories moved into feature/backend layers. The unused
  `library/backends/` documentation/compatibility tree was removed.
- Root `include/api` and `src` moved to `library/include/api` and `library/src`.
- CUDA attention implementation ownership moved from `API.cpp/cuda/attention/*` to each feature's `cuda/translation/gpu`.
- The stable-diffusion GGML xFormers adapter converts GGML tensors to the
  Common request and never calls the CUDA translation directly. Remaining
  direct Flash/Sage source shims are routing debt, not a permitted pattern.
- The old materialized xFormers CUDA stage files moved from
  `cuda/translation/gpu/{qkt,mask,softmax,av}.cu` to `prototype/cuda-*.cu` after
  fused `forward.cu` replaced them as the registered Common translation.
- `source/API.test/Feature/Attention/Xformers/Cpu/Main.cpp` was replaced by the
  Common cycle test under `source/API.test/MultiTest/xformers/cycle/`.
- `IMPLEMENTATION_STATUS.md` is the authoritative implementation and
  validation ledger. `TODO_GRID.md` is retained as a legacy planning grid.

## Adding a backend implementation

1. Document the backend API in `definition/[device-type]/[method].cpp`.
2. Implement its adapter in `translation/[device-type]/[method].cpp`.
3. Compare translations and extend feature `common/` only for genuinely shared behavior.
4. Register the translation with Common and wire both into the build.
5. Test compile-time selection, runtime capability checks, numerical behavior, and fallback behavior.
6. Expose only Common to Easy Diffusion.

## Validation and GitHub Actions

GitHub Actions supplies cross-backend compiler and toolchain coverage only.
Its compiler workflows can establish source selection, compilation, and linking
for the represented targets. Runtime capability,
numerical correctness, fallback behavior, memory behavior, and performance need
separate validation on real devices. An `IMPLEMENTATION_STATUS.md` backend cell
remains blank until the complete source route is wired, its applicable compiler
workflow passes, and required runtime validation is recorded. Workflow
existence alone is not evidence of support.
