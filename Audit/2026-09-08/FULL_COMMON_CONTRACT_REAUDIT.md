# Full Common contract re-audit

Date: 2026-09-08  
Audited Theory commit: `ac0f52bb8f8583553b41974004c9bd88e8e4490c`  
Audited tree: `7f77e7edd656facd237ed0571a283cdeec4f658c`

## Verdict

Theory is not yet conformant with the newly clarified Common contract.
Detection and model lifecycle have the intended layering, but every currently
executable attention family needs at least one ownership correction. Earlier
runtime successes remain useful behavioral evidence; they are not proof that
the current definition/translation/Common topology is correct.

The required communication route is:

`native definition -> translation -> Common -> translation -> native definition`

Common is the shared message contract. A definition may use native driver
language only for work wholly inside itself. The rule applies to same-device
work as well as cross-device work. There is no direct definition-to-definition,
backend-to-backend, translation-to-translation, or caller-to-definition route.

## Re-audit matrix

| Area | Topology | Behavior evidence | Result |
| --- | --- | --- | --- |
| Library entrypoints | Library calls Common only | Static inspection | Pass |
| Detect | Common dispatches translations; translations adapt definitions | CPU runtime passed now; other backends not rerun | Pass for inspected topology |
| Model load/unload | Common registry -> typed translation -> definition | CPU runtime passed now; CUDA not rerun | Pass for implemented model routes |
| Other load/unload resource types | Common enum exists; translations and definitions absent | None | Incomplete |
| Overflow CPU | Common dispatch -> translation -> definition | CPU runtime passed now | Pass for current CPU allocation/release |
| Overflow CUDA | Translation calls Common planning after dispatch | Historical CUDA evidence only | Topology fail |
| xFormers CPU | Compute lives in translation; CPU definition is absent | CPU semantic and 32-cycle test passed now | Topology fail |
| xFormers CUDA | Kernels and CUDA runtime calls live in translation | Historical runtime/application evidence | Topology fail |
| Flash CPU | Entire attention implementation lives in translation | CPU runtime passed now | Topology fail |
| Flash CUDA | Fused kernel and CUDA execution live in translation | Historical runtime/application evidence | Topology fail |
| Sage CUDA support | Common support request -> translation -> definition | Historical support test | Pass for support only |
| Sage forward | No complete Common forward route | Historical native-kernel evidence | Incomplete |
| Flex CPU | Algorithm lives in Common header; no definition or registered translation | Source only | Topology fail/incomplete |
| Split CPU | Algorithm is explicitly a prototype; Common files are placeholders | Prototype evidence only | Incomplete |
| Primitives | Layout documented; no implementations | None | Incomplete |
| INFERENCE tokenization | Owned implementation exists in INFERENCE | Direct manual build/run passed | Partial feature |
| INFERENCE VAE | Encode/decode planning exists; execution does not | Direct manual build/run passed | Planning only |
| INFERENCE image generation | Stage plan only; no denoiser, sampler execution, VAE execution, or result assembly | None | Not capable |
| UI parity with main | Entire tracked `ui/` tree matches current `origin/main` | Git tree comparison | Pass |
| Report-file privacy | Python report handler redacts prompt context and hashes/redacts paths | Four tests passed now | Pass within handler scope |
| Native/console logging privacy | C++ logger prints formatted messages unchanged; call sites log prompts and paths | Static inspection | Coverage fail |
| SDKIT3/llama independence | No prohibited production includes or exact DiffUser source copies detected | Boundary script passed now | Pass within current scanner scope |

## Blocking findings

### C-01: attention execution is owned by translations

The following files perform mathematical or device execution instead of only
adapting Common values to definition-native values:

- `features/attention/xformers/cpu/translation/cpu/qkt.cpp`
- `features/attention/xformers/cpu/translation/cpu/mask.cpp`
- `features/attention/xformers/cpu/translation/cpu/softmax.cpp`
- `features/attention/xformers/cpu/translation/cpu/av.cpp`
- `features/attention/xformers/cpu/translation/cpu/forward.cpp`
- `features/attention/flash/cpu/translation/cpu/flash_attention_cpu.cpp`
- `features/attention/xformers/cuda/translation/gpu/forward.cu`
- `features/attention/flash/cuda/translation/gpu/forward.cu`

The CPU files contain the attention loops directly. The CUDA translation files
contain `__global__` kernels, kernel launches, pointer inspection, device
selection, and synchronization. These implementations must move into matching
definitions. Their translations should be limited to conversion, definition
invocation, and conversion of the result back to Common.

### C-02: Flex places native work in Common

`features/attention/flex/common/flex_attention.hpp` contains selection,
normalization, thresholding, pooling, and expansion. Common should define and
validate the normalized message contract, not execute the CPU algorithm. Move
the host algorithm into a CPU definition and add a CPU translation and Common
registry/dispatcher.

### C-03: CUDA Overflow reverses the layering mid-route

`features/overflow/cuda/translation/gpu/allocate.cpp` calls the Common `plan`
function after Common has already selected the CUDA translation. Planning and
cross-backend candidate selection belong before backend dispatch. The
translation should receive a normalized selected request and invoke only its
CUDA definition.

### C-04: the current boundary tests cannot prove the new contract

`scripts/test_api_boundaries.py` proves separation from SDKIT3 and llama.cpp,
but it does not detect algorithms or driver calls inside translations, native
work inside Common, reverse calls from translation into Common, or missing
definitions. `scripts/test_inference_boundaries.py` correctly rejects direct
INFERENCE access to definitions/translations, but it cannot establish that the
Common surface it accepts is itself correctly layered.

`source/API.test/Feature/Attention/Sage/Cuda/Support.cpp` also includes a
translation header and calls `ensure_registered()` directly. Registration may
need a static-link anchor, but API.test should exercise the public API surface;
an internal registration test must be clearly separated from public contract
tests.

### C-05: privacy-safe reporting does not cover every log sink

`ui/easydiffusion/privacy_debug.py` correctly protects the rotating report file:
existing files become content MD5 identifiers, unreadable files receive
identifier-preserving alphabetic redaction, and prompt-like messages are
blocked. The current file matches `main`, and all four privacy tests pass.

That handler explicitly leaves console logging unchanged. In addition,
`source/sdkit3-port-source/src/logging.cpp` writes the fully formatted C++
message without sanitization. Current C++ call sites include raw prompts, model
paths, VAE paths, adapter paths, and model directory paths. Therefore a report
assembled from console/native output, or any independently captured native
log, can still expose the data the Python report handler protects.

No suspicious tracked filename was identified by the audit's lexical filename
scan. The two filenames containing `nsfw` are functional checker modules, not
user model or generated-content names. This is not a content-classification
guarantee.

## Non-blocking but incomplete areas

### Typed lifecycle registry

The load/unload Common tables are correctly keyed by backend and resource type,
preventing a VAE or other resource from silently borrowing the model route.
Only `model` has CPU and CUDA definitions/translations. `vae`, `unet`, `clip`,
`clip_vision`, `condition`, `image`, `latent`, and `mask` are contract labels,
not implemented support.

### INFERENCE.cpp

INFERENCE currently contains a generation request validator, stage planner, a
self-owned BPE tokenizer, and VAE shape/planning logic. It does not perform
conditioning, latent initialization, sampling/denoising, VAE encode/decode, or
result assembly. It therefore cannot generate an image when linked into
API.test today. This is an honest foundation rather than a failed generator.

The production includes pass the Common-only boundary scan. Tokenization is
allowed to remain owned by INFERENCE. VAE execution should also remain owned by
INFERENCE, while DiffUser owns preparation, placement, handles, and lifecycle.

### CPU attention accounting

- xFormers: behavior exists, but it must be moved from translation to a CPU
  definition before it is conformant.
- Flash: behavior exists, but it must be moved from translation to a CPU
  definition.
- Flex: behavior exists in the wrong layer and lacks registration.
- Split: prototype only; no CPU definition or translation.
- Sage: no CPU implementation.

Thus no CPU attention family currently satisfies all of definition ownership,
translation-only adaptation, Common dispatch, public API testing, and runtime
validation at once.

## Validation executed during this audit

Passed:

- CPU detection runtime test.
- CPU model load runtime test.
- CPU model unload ownership and double-unload test.
- CPU overflow allocation/release and mixed-backend plan test.
- CPU xFormers semantic suite and 32 lifecycle cycles.
- CPU Flash behavioral test.
- Three INFERENCE tests compiled directly with C++17 and ran successfully.
- Four privacy-debug tests.
- DiffUser/SDKIT3/llama boundary scan.
- INFERENCE dependency boundary scan.
- Main-to-Theory `ui/` equality check.

Not executed:

- CMake/CTest, because CMake is unavailable in the audit environment.
- CUDA, ROCm, oneAPI, OpenCL, OpenVINO, OpenGL, Vulkan, Mesa, and DirectML
  runtime tests because their required runtimes/hardware were unavailable or
  were outside this environment.

Passing behavior tests for C-01 routes must not be promoted to topology proof.

## Correction order

1. Extend automated boundary checks to reject driver APIs and kernels in
   translations, algorithms in Common, translation-to-Common reverse calls,
   direct test includes of internal layers, and executable routes without a
   definition.
2. Move CPU xFormers and CPU Flash execution into CPU definitions without
   changing their Common contracts or behavioral vectors.
3. Move CUDA xFormers and CUDA Flash kernels/runtime work into GPU definitions;
   leave translations as adapters.
4. Move CUDA Overflow planning ahead of backend dispatch.
5. Split Flex into Common contract, CPU translation, and CPU definition.
6. Implement Split through the same layers, using the prototype only as a
   semantic reference.
7. Add Sage CPU only after the reusable attention/primitives contracts exist.
8. Extend privacy sanitization to native and captured console logging before
   treating generated diagnostic bundles as safe to share.
9. Re-run backend hardware tests, then rebuild the status ledger from the new
   topology rather than inheriting earlier maturity marks.

## Final assessment

The recent merge did not contaminate DiffUser with a production dependency on
SDKIT3 or llama.cpp according to the current static and exact-copy checks. The
main invasive result is architectural: working attention code was placed in
translations (and Flex in Common), making the differences appear smaller while
violating the intended ownership boundary. The correct response is relocation
and stronger contract tests, not deletion of the working algorithms.

