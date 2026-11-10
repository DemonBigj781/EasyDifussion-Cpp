# DiffUser.cpp implementation and validation status

This is the authoritative implementation and validation ledger.
Directory presence, a placeholder, a design document, or a compiler workflow
does not establish backend support. See the
[dated layout audit](../../Audit/2026-09-06/LAYOUT_AUDIT.md) before interpreting
a path.

The [full Common contract re-audit](../../Audit/2026-09-08/FULL_COMMON_CONTRACT_REAUDIT.md)
supersedes topology conclusions in the older audits. Existing runtime marks
record behavioral history only where that re-audit identifies execution in the
wrong layer; they must not be read as definition/translation/Common conformance.

The post-audit ownership correction moves CPU/CUDA xFormers and Flash execution
into definitions, moves Flex CPU execution out of Common, and moves CUDA
Overflow planning ahead of dispatch. Current post-correction validation reaches
runtime-tested maturity for the CPU/CUDA xFormers, FlashAttention, and Overflow
routes and for CPU Flex selection. See the
[ownership-correction record](../../Audit/2026-09-08/OWNERSHIP_CORRECTION.md)
for the exact source moves, commands, hardware, and end-to-end boundary.

`TODO_GRID.md` is retained as a legacy planning grid and must not be used for
current support or validation claims.

## Status legend

| Mark | State | Meaning |
| --- | --- | --- |
| blank | Not established | No implementation or validation state has been established. |
| `I` | Implemented | Source exists but may still contain incomplete behavior. |
| `C` | Code complete | The intended contract is implemented with no known required source path missing. |
| `B` | Builds | The applicable backend/toolchain compiled it; runtime is not implied. |
| `R` | Runtime tested | Relevant runtime tests executed on the applicable runtime/hardware. |
| `E` | End-to-end tested | Easy Diffusion/application -> Common -> translation -> definition -> native runtime/device executed. |
| `N/A` | Deliberately not separate | The method is fused into another operation or deliberately has no separate callback. |
| `XXX` / `XXXX` | Proven complete | Intended paths, integration, edge/error behavior, ownership/cleanup/fallback behavior, and required validation have been accounted for. |

Status is progressive: `I -> C -> B -> R -> E -> proven complete`. A later
state includes earlier expectations. `N/A` is not a maturity state.

The earlier mixed-case grid used `XXX`/`XXXX` for Detect, lifecycle, and Overflow
before the Common-routing audit exposed missing revalidation or direct-routing
debt. This replacement preserves the runtime history in prose but records the
currently supportable maturity instead of carrying those proof marks forward.

## Detect

`detect` reports backend availability and runtime-visible devices. It does not
mean model-format, ControlNet, or image-object detection.

| Method | CPU | CUDA | ROCm | oneAPI | OpenCL | OpenVINO | OpenGL | Vulkan | Mesa | DirectML |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `detect` | `R` | `R` | `I` | `I` | `R` | `I` | `R` | `R` | `R` | `I` |

Source routes and compiler workflows exist for every column. CPU, CUDA,
OpenCL, OpenGL, Vulkan, and Mesa have current local runtime evidence. The
remaining columns are source-audited only.

## Load and unload

The current normalized contract owns an opaque model-byte resource. It does not
claim model parsing, tensor construction, or graph construction.

| Component/method | CPU load | CUDA load | CPU unload | CUDA unload | Other backends |
| --- | --- | --- | --- | --- | --- |
| `model` | `R` | `R` | `R` | `R` |  |
| `clip` |  |  |  |  |  |
| `clip-vision` |  |  |  |  |  |
| `condition` |  |  |  |  |  |
| `image` |  |  |  |  |  |
| `latent` |  |  |  |  |  |
| `mask` |  |  |  |  |  |
| `unet` |  |  |  |  |  |
| `vae` |  |  |  |  |  |

Load validates and copies bytes into backend-owned storage. Unload releases
through the owner and clears the normalized resource only after success.
The Common registry is keyed by both backend and resource type so future tensor
types cannot silently borrow the model route. Only the model entries are
currently registered; blank rows remain unsupported.

## Overflow

| Method | CPU | CUDA | ROCm | oneAPI | OpenCL | OpenVINO | OpenGL | Vulkan | Mesa | DirectML |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `plan` | `R` | `R` |  |  |  |  |  |  |  |  |
| `allocate` | `R` | `R` |  |  |  |  |  |  |  |  |
| `release` | `R` | `R` |  |  |  |  |  |  |  |  |

After the ownership correction, CPU allocation, ownership rejection, release,
and a synthetic mixed-backend Common plan passed locally. CUDA validation on an
RTX 3060 (`sm_86`, driver 580.94.18, CUDA toolkit 12.4) covered the regular
forced system-RAM route and a bounded real VRAM OOM. The stress test recorded a
2048 MiB CUDA-managed fallback, verified GPU access, and released the fallback
and reservations. The mapped-host path builds as the fallback when concurrent
Managed Memory is absent.

CUDA secondary placement covers runtime-visible CUDA GPUs only. Mixed-vendor
secondary devices, `gpu-zram`, zram, and swap remain unimplemented. H3C XG310
validation must represent four independent 8 GB Intel SG1 devices, not one
contiguous 32 GB allocation. CPU/CUDA Library handlers now route through Common.

## Attention

| Feature/method | CPU | CUDA | ROCm | oneAPI | Other backends |
| --- | --- | --- | --- | --- | --- |
| xFormers feature | `R` | `R` |  |  |  |
| xFormers `qkt` | `R` | `N/A` |  |  |  |
| xFormers `mask` | `R` | `N/A` |  |  |  |
| xFormers `softmax` | `R` | `N/A` |  |  |  |
| xFormers `av` | `R` | `N/A` |  |  |  |
| xFormers `forward` | `R` | `R` |  |  |  |
| Sage feature |  | `R` |  |  |  |
| Sage `support` |  | `R` |  |  |  |
| Sage `forward` |  | `R` |  |  |  |
| FlashAttention | `R` | `R` |  |  |  |
| Flex Attention | `R` |  |  |  |  |
| Split Attention | `I` |  |  |  |  |

xFormers CPU now uses registered materialized adapters backed by CPU
definitions. CUDA registers only a fused `forward` adapter backed by its GPU
definition; QKT, mask, softmax, and AV semantics execute inside that native
kernel. Both corrected routes pass their Common semantic suites and 32
model-byte lifecycle cycles. The CUDA path also passes Compute Sanitizer
memcheck with zero errors on the RTX 3060.

The historical pre-isolation stable-diffusion.cpp adapter completed a 512x512
one-step generation with 20 Common CUDA xFormers launches. Commit `d883d6e`
removed that GGML bridge while isolating DiffUser from the engine. The current
snapshot therefore has no application adapter or launch-counter symbol and
does not retain `E` maturity. The stale generator builds the full current
stable-diffusion/ggml CUDA libraries, but cannot link its old observation hook.

Sage CUDA now registers a normalized Common `support` translation and uses it
from stable-diffusion.cpp's GGML selector. Its build-only test covers
registration, SM86 acceptance, Hopper rejection, F32 Q/output with F16 K/V,
head dimensions 64/128, grouped-query-compatible shapes, GGML output strides,
and rejection of unsupported masks and dtypes. On an RTX 3060 (`sm_86`, driver
580.94.18), a profiled one-step 256x256 SDXL Turbo generation completed and
wrote a valid RGB PNG. Its Nsight Systems trace recorded 280 Sage key-mean
launches, 280 F16 K quantization launches, 280 F32 Q quantization launches, 280
INT8-QK/FP16-PV attention launches, and 280 output-conversion launches. The
static-library registration anchor was exercised by this application build, so
the live selector entered Common before the native launches. Sage therefore has
CUDA runtime evidence, but remains below `E`: forward still calls the
GGML-native SM80 compatibility kernel rather than a normalized Common `forward`
request, and numerical parity plus Compute Sanitizer coverage remain open.

FlashAttention CUDA registers a separate fused `forward` translation behind
the normalized Common contract, with native execution owned by its CUDA
definition. The online-softmax kernel supports F32, F16,
and BF16 Q/K/V inputs, F32 output, additive masks, GGML-style max-bias/ALiBi
mask scaling, logit soft-capping, grouped-query attention, byte strides, and an
opaque CUDA stream. After relocation, an RTX 3060 (`sm_86`, driver 580.94.18,
CUDA 12.4) passed deterministic numerical tests and Compute Sanitizer memcheck
with zero errors. The Flash MultiTest also completed 32 model-byte load/return,
Flash forward/result, and model-unload API roundtrips through Common under
memcheck.

The historical pre-isolation GGML application adapter completed a 512x512
one-step SD 1.5 generation in 26.7 seconds with 40 normalized Common Flash
launches. That adapter and counter were removed by `d883d6e`, so the current
route is `R`, not `E`. The upstream optimized GGML `fattn` implementation is an
engine path and does not establish Common validation. The Nouveau Quadro K2000
is not a CUDA device, and its Kepler `sm_30` architecture is below the Common
route's Pascal minimum.

The planned normalized method inventories are:

- Sage: `support`, `key_mean`, `quantize`, `qkt`, `mask`, `softmax`, `sv`,
  `normalize`, and `forward`.
- Flash: `support`, `workspace`, `select`, `qkt`, `mask`, `softmax`, `av`,
  `combine`, and `forward`.
- Flex: `mask_size`, `validate`, `sum_heads`, `margin`, `normalize`,
  `threshold`, `pool`, `expand`, and `select`.
- Split: `plan`, `split`, `qkt`, `mask`, `softmax`, `av`, `merge`, and
  `forward`.

Blank feature rows remain blank even where source or prototypes exist; build
and runtime maturity must be established explicitly.

Detailed CPU family and semantic coverage is recorded in the
[dated CPU attention audit](../../Audit/2026-09-08/CPU_ATTENTION_AUDIT.md).
Post-correction xFormers and Flash tests re-establish the runtime marks above;
they do not add coverage for every open capability listed by that audit.

### CPU Flex, Split, and Sage accounting

Flex CPU is `R`. Common owns the normalized contract and registry, the CPU
translation adapts requests and results, and the CPU definition owns the
selection algorithm. A public Common-only API.test covers translated
capabilities, deterministic 2x2-to-4x4 selection and expansion, compatibility
wrappers, insufficient capacity, and rejection of an unregistered backend.

Split CPU is also `I`, not `B` or `R`. A backend-neutral exploratory prototype
implements planning and forward execution, including exact query and K/V split
behavior. The files under `split/common/` remain structural placeholders;
there is no registered CPU translation and no application-facing Common route.
The `I` marks account for existing source only and must not be interpreted as
SDKIT3 integration or runtime support.

Sage has a Common support/capability registry but no CPU definition,
translation, forward route, or test. Its CPU status therefore remains blank.

## Cache

`features/cache/` is currently a `.gitkeep`-only scaffold. EasyCache and
TeaCache are planned inventories, not implemented source families.

| Cache | CPU | CUDA | ROCm | oneAPI | OpenCL | OpenVINO | OpenGL | Vulkan | Mesa | DirectML |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EasyCache |  |  |  |  |  |  |  |  |  |  |
| TeaCache |  |  |  |  |  |  |  |  |  |  |

Planned EasyCache methods: `support`, `validate`, `reset`, `init`, `enabled`,
`sigma`, `begin_step`, `active`, `skipped`, `has_cache`, `store`, `apply`,
`before`, and `after`.

Planned TeaCache methods: `support`, `configure`, `validate`, `init`, `enabled`,
`begin_step`, `rel_l1`, `rescale`, `accumulate`, `reuse`, `store`, `apply`,
`skipped`, `before`, and `after`.

## Primitives

Project-wide primitive requirements are classified in the dated
`Audit/2026-09-08/INFERENCE_PRIMITIVE_AUDIT.md`. No backend/device/type primitive
source has yet passed the Common contract, implementation, build, and runtime
gates. Existing `Operation` enum members are inventory labels, not primitive
support claims.

## Deferred feature scaffolds

`ggml`, `gguf`, and `reserve` are explicitly later-development directories.
Other empty top-level families are classified in the
[dated layout audit](../../Audit/2026-09-06/LAYOUT_AUDIT.md). They are not added
to this ledger until a normalized method contract exists.
