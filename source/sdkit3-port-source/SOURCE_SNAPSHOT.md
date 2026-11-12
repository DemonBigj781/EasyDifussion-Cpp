# sdkit3 source snapshot

This directory is a de-gitted snapshot of the locally patched native sdkit3
source used by EasyDifussion-Cpp.

- Snapshot date: 2026-09-05
- sdkit upstream: <https://github.com/easydiffusion/sdkit>
- sdkit base revision: `de8ff820431358343338ee4363a5fb998e8240a4`
  (`v3.4.1`)
- stable-diffusion.cpp upstream: <https://github.com/leejet/stable-diffusion.cpp>
- stable-diffusion.cpp base revision:
  `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`
- ggml base revision: `e20c3a14aa70ee84ca58499814206dd08d8026bc`

The snapshot includes the upstream stable-diffusion.cpp and GGML changes through
the revisions above together with the local native backend changes present in
the source checkout at refresh time, including the ControlNet, LLLite,
IP-Adapter scheduling, latent-interposer, Flex Attention, TeaCache, xFormers,
SageAttention, image-tool, and vision extensions.

Nested `.git` metadata, generated CMake build directories, Python caches,
runtime `options.json`, editor settings, and private credential files are
intentionally excluded. The original source checkout and its Git history are
not modified by this snapshot.

The nested CMake projects use the recorded snapshot revisions as build-info
fallbacks when their own `.git` metadata is absent. This prevents them from
misidentifying the parent EasyDifussion-Cpp revision as their upstream commit.

## Selective SDKit v3 reconciliation (2026-11-11)

The application layer selectively incorporates GPT-OSS/MoE tensor detection
from upstream `35d53a696fe151ebf98e072adaeac7f9444388b9` and LLM-vision
projector detection/loading from `22b4cbc8fe63e588a791fe10a1ea3f22df0e656d`.
The model-detection tests come from SDKit v3 at
`9679c11096f167dc6aae2c131dfa702d6dd21441`, excluding the tokenizer-JSON test.
Vision projector paths participate in this fork's model-configuration key
and reload tracking, separately from its existing CLIP-vision/IP-Adapter path.

This is not a wholesale upstream refresh. External tokenizer JSON loading
is not exposed: the preserved diffusion API has `llm_vision_path`, but lacks
the newer tokenizer parameter/parser. That part requires a separate engine
reconciliation. Detection tests and compilation do not establish end-to-end
generation or model-quality coverage.


## Selective SDXL textual-inversion reconciliation (2026-11-11)

The custom CLIP conditioner adapts upstream stable-diffusion.cpp
`d04e8950c1ec8d30248cbe996682b3182fb1adf6` (SDXL embeddings loading).
CLIP-L and CLIP-G now have separate custom-embedding buffers with shared token
IDs, missing rows are zero-padded, source tensors are converted to the destination
embedding type, failed loads do not publish partial state, and temporary loading
contexts are freed. Row storage uses `ggml_row_size` instead of assuming scalar
storage. The fork's independent CLIP-L/CLIP-G chunking and training prompt
truncation are retained.

The CPU-only `textual-inversion-test` target (with `SDKIT_BUILD_TESTS=ON`)
generates tiny synthetic safetensors in a temporary directory. It checks paired,
one-sided and unequal-row SDXL embeddings, cache reuse, append offsets, F16 input
conversion, failure state preservation and single-encoder SD1 compatibility.
No model downloads are needed. This verifies loader state, not GPU execution,
full text-encoder inference, image quality or portable builds.


## Selective K-quant embedding reconciliation (2026-11-11)

The embedding dtype allowlist adapts upstream stable-diffusion.cpp
`9cdb6b6fc05edf2d446bfe881a4c462b241b3025` (preserve K-quantized embedding
weights) into this fork's existing `src/core/ggml_extend.hpp`. Q2_K through Q6_K
remain quantized instead of being expanded to F32 during embedding initialization.
Existing supported types, unsupported-type fallback, and forced-F32 CLIP loading
for textual inversion are unchanged. No GGML source or dependency revision changes.

The CPU-only `embedding-quantization-test` target checks actual synthetic quantized
row gathering, reordered/repeated row IDs, CLIP-L and CLIP-G weight types, and their
forced-F32 override. CPU operations are tested against the preserved GGML backend;
CUDA/Vulkan support is present in source but is not runtime-validated by this test.


## Selective GGML logging reconciliation (2026-11-11)

The GGML log callback adapts the literal-text safety fix from upstream
`b8248a869cc02169c494b2a36f59f34979e7dd96` (#2002) at its location in this fork,
`src/core/ggml_extend.hpp`. Incoming GGML log text is passed as a string argument,
never as the printf format. Existing severity mapping, source annotations, newline
handling and formatted application logs remain unchanged.

This is not the upstream logging rewrite: fixed-buffer truncation and shared-buffer
thread-safety behavior are unchanged. The `ggml-log-forwarding-test` target covers
literal format directives, severity routing, empty/Unicode/newline text, ordinary
formatted logging and a missing callback. No backend/dependency update is needed.


## Selective vision-projector compatibility reconciliation (2026-11-11)

The LLM config detector adapts upstream `ca37fad89a06b8a51344c1e631ab0fd9a2f0c230`
(#1918). Vision is disabled before LLM model construction if the detected projector
output dimension differs from the LLM hidden size. Matching projectors remain
available; absent vision weights retain existing disable behavior. Undetected
projector dimensions are not compared against architecture defaults.

The metadata-only `vision-projector-compatibility-test` covers both supported merger
weight names, matching/mismatching dimensions, caller-disabled and missing vision,
undetected defaults and unrelated tensor prefixes. This is not actual multimodal
model loading or inference validation; no tokenizer, pipeline or GPU API changes.


## Selective CLIP crop-bounds reconciliation (2026-11-11)

CLIP preprocessing adapts upstream `d32b4e893bc429cae6bcfc6297db76c46ad2ed82`
(#1995), clamping resized dimensions to at least the center-crop target. This
avoids a one-pixel shortfall from float rounding, while preserving existing
truncation for dimensions that already cover the crop window.

The CPU-only `clip-preprocess-bounds-test` uses synthetic tensors covering square,
portrait, landscape, rectangular and identity resizing, ordinary center-crop
position, input clamping and CLIP channel normalization. The unpatched 23-to-224
case throws an out-of-range tensor-index error. No image/model files or GPU needed.


## Selective BF16 embedding reconciliation (2026-11-11)

The embedding allowlist and CLIP token-weight operation annotation adapt upstream
`e95ab9699755717877018e4334ab64b10e669479` (#1959). BF16 embedding tensors remain
BF16 rather than expanding to F32, and backend planning sees the token weight's
GET_ROWS operation. CLIP position weights and forced-F32 textual-inversion loading
retain their existing behavior. No GGML dependency change is needed.

The CPU-only `bf16-embeddings-test` checks exact scalar-reference row gathering,
compact storage, CLIP-L/G dtype and operation annotations, position weights and
forced-F32 behavior. The earlier K-quant test's unsupported-type check now uses
I32/I64, since BF16 support is explicitly added by this subsequent reconciliation.


## Selective prompt-parser stack-safety reconciliation (2026-11-11)

The parser adapts bounded plain-text scanning and separate numeric-weight scanning
from upstream `4a7da26b73138e6ee5b32256ab93cb6dd66d75f4` (#1919). Unbounded
libstdc++ regex recursion on long plain or numeric runs is avoided. The fork's
std::stof conversion and exception behavior are deliberately retained, unlike the
upstream literal-text fallback, because native request admission relies on those
errors to reject invalid or overflowing attention weights.

The `prompt-attention-bounds-test` covers long plain/Unicode/numeric runs, attention
and BREAK semantics, numeric exceptions and native admission status. This does not
change the fork's quotation/concept parser, rewrite prompts or impose a new length
limit. Existing parser quirks and admission semantics remain outside this fix.

### Single-output linear LoRA merge

Adapted leejet/stable-diffusion.cpp #2116 (`f2f168fa70c812a689c8ec2cb51b62708a556723`) in the fork-local `core/ggml_extend.hpp` helper. Preserve the singleton output dimension when GGML reports a rank-by-one linear LoRA up tensor as one-dimensional. CPU regression covers rank-one/rank-three linear merges, singleton/multiple outputs and ordinary convolution controls against scalar multiplication. No adapter loading, scaling, training, or dependency changes.

### Integer LoRA scalar loading

Adapted leejet/stable-diffusion.cpp #2263 (`b4d6b83af6d12e5770306f82b1e5625ed20ec237`) to convert I32 tensors directly to F32 in the existing model loader. This also supports ordinary I64 LoRA alpha values through the loader's existing I64-to-I32 path; it does not widen or change that narrowing contract. Regression covers scalar safetensors loading with/without mmap, both integer source formats, multirow conversion including rounding/extremes, destination bounds and existing I32-copy/F16 behavior.

### Standalone SDXL UNet classification

Completed the missing embedding-signature portion of leejet/stable-diffusion.cpp #2247 (`e7b7ec6d6f6eed6c005a0ccd7b811dc3d9391540`). The fork already recognizes prefixed diffusers block names and input shapes; preserve that existing work. UNets carrying SDXL added text/time embeddings now identify as SDXL without requiring an accompanying CLIP-G checkpoint. Preserve input-channel inpainting/pix2pix and SSD/Vega topology classification, video/model-family precedence and custom family detection. Tests use synthetic tensor metadata, not model inference.

### Custom sigma schedule step counts

Adapted leejet/stable-diffusion.cpp #2084 (`c150a6bafeb21824f30b813ae1212321a5d131fd`) in the fork's existing SamplePlan. A longer custom sigma list now raises the single-model sample-step count as well as total steps; explicit dual-model partitions retain the low-noise allocation and adjust the high-noise remainder. Extracted only this count adjustment into `runtime/custom_sigma_steps.h` for direct CPU regression. Preserve automatic-boundary negative sentinel, custom sigma values, sampler arguments, image/video constructors and fork-specific hires paths.

### LLM attention execution preferences

Adapted leejet/stable-diffusion.cpp #1987 (`269e726015184977f363c838f9082f0edaf640f9`) to the fork's older attention-helper signature. Forward the complete fork-local flash/Sage preference bit field, plus its training-graph exclusion, rather than replacing the custom interface with upstream's boolean. CPU regression executes a small production LLM Attention block with masked/unmasked inputs, checks selected graph operations and Sage flags, and compares numerical output against the ordinary path. No GPU acceleration or full model inference is claimed.

### Requested hires sampling steps

Adapted leejet/stable-diffusion.cpp #2227 (`9bfc4d108916c5cd3f0cd7c8206b41f169e33cd6`) in the existing fork hires sigma resolver. Select the schedule tail from the requested effective step count, rather than multiplying the rounded scheduler count by denoising strength again. A small private index helper permits direct regression of 700 step/strength combinations and boundary behavior. Custom sigma bypass, scheduler selection, strength handling, image/video callers and default-step resolution remain unchanged.

### CPU FP8 linear fallback

Adapted leejet/stable-diffusion.cpp #2140 (`2891378bb17dbed110b391f9aab47d241413153d`) in the fork-local Linear implementation. CPU backends use the existing BF16 cast fallback for FP8 weights even if their support query accepts direct matrix multiplication. Preserve other backend choices, forced precision, bias, weight scaling and custom integer paths. Regression inspects and executes synthetic E4M3/E5M2 linear graphs, with BF16/F32 controls.

### SD3 encoder chunk alignment

Adapted leejet/stable-diffusion.cpp #2111 (`e16d26a96a1bf59824d12cc26352aa36c25c2252`) to the fork's concrete tokenizer members. Before conditioning slices the longest token stream, pad each shorter active CLIP-L/CLIP-G/T5 stream with its tokenizer's complete empty chunks. Preserve weighted prompt prefixes, distinct BOS/EOS/padding IDs, disabled encoders and existing tokenization/chunking/truncation. Regression exercises production padding with tokenizer configurations matching these three families, without model/vocabulary loading.

### Short Wan decode-plan tails

Adapted leejet/stable-diffusion.cpp #2197 (`4c870cc60cd466427e4802d35a15b681ba9e4e90`) in the fork's stateful Wan decode path. Balance a singleton final latent tile only for eligible two/three-tile, non-2D Wan plans. Preserve total coverage, tile metadata, cache handling, long plans and other model families. Regression exercises the production plan helper and tile slicing/concatenation over 1,584 cases; actual VAE decoding and memory-peak improvement require model-level validation.

### Linux physical-core topology path

Adapted leejet/stable-diffusion.cpp #2186 (`ddd6b3aa3574ecc9d1ebb7dd92e726a53ba3bb3b`): scan `/sys/devices/system/cpu/cpuN/topology/thread_siblings`, not `/sys/devices/system/cpuN/...`. Extracted the existing scan into a private helper for temporary-filesystem tests covering mixed SMT, duplicated masks, missing/empty files and the existing first-missing-CPU stopping rule. Preserve platform fallbacks and explicit thread settings. This does not add CPU affinity/cgroup awareness or sparse-ID enumeration.

### VAE retry prerequisites: runner status and cache ownership

Adapted the status-reporting prerequisite of leejet/stable-diffusion.cpp #2019 (`97d932b8f81264165cee0285fd62484ce67952d6`) to the fork's templated GGMLRunner. Report proven graph/cache allocation failures, cancellation and successful completed output conservatively; ambiguous weight preparation and readback failures remain FAILED. Reset status per compute/segment and retain it through cleanup. Cache-buffer allocation now returns failure instead of asserting, with scoped cleanup of old/replacement cache ownership. Linux tests inject allocation failures at link time into actual tiny CPU graphs and inspect ownership/status recovery. Retry policy, non-finite pixel recovery and VAE tiling choices are deliberately unchanged. This is not the complete upstream retry integration.

### VAE decode retry eligibility

Continue the adapted #2019 retry reconciliation with a separate per-decode outcome. Tiling retries now require a confirmed allocation failure or the fork's non-finite-pixel recovery case; cancellation and other failures do not trigger repeated decode. Reset the outcome and runner status before each base decode to avoid inheriting a previous failure, retain the failed tile's outcome, and recheck cancellation before each auto-fit/image fallback. Preserve temporal-first auto-fit, legacy automatic image tiling, existing tile dimensions, and release the compute buffer before the image fallback as well as auto-fit. CPU tests inject reserve/readback failures into actual small VAE graphs, cover spatial/temporal tile failure retention, NaN/Inf recovery, cancellation and subsequent success. Scheduler/multi-device failure injection and model-weight inference are not covered. Ambiguous weight-preparation failures remain non-retryable FAILED; allocator/assert paths outside the conservative runner-status implementation are not claimed as classified. This does not implement upstream tile halving or a full retry-policy merge.

The decode-first-stage retry loop is now a shared internal helper used by the production callsite and CPU regression tests. Tests execute that same orchestration with actual small VAE graphs: successful decode, persistent allocation failure, non-finite recovery, non-retryable failures, cancellation between attempts, temporal-before-spatial ordering, maximum attempt counts, auto-fit-off video behavior, configured image tile sizes/overlap, and preservation of caller settings for the temporary image fallback. Latent conversion and backend selection remain in decode_first_stage and still require model-level integration validation.

### Bounded circular/non-circular spatial tiling

Adapted leejet/stable-diffusion.cpp #2088 (`019bcfc0ae3bc1a5bf37b93c65727dee149a9c31`) and #2105 (`fb8ee71f55ad603dc0e567984663c147c71879af`) using the bounded axis-span planner and per-edge blending from upstream runtime/tiling.cpp at `32d516c0197ed5ca31305b8575e14eb1451fad33`. Replace the old negative-overlap-prone formula and tail repositioning inside the fork's existing templated spatial tiler. Preserve all VAE tile-size conversion and latent/small-dimension units; do not import #2059's image-pixel configuration change. Preserve arbitrary higher tensor planes (including video), encode/decode scale mapping, circular wrap flags, and immediate empty-result propagation. Tile placement and seam blending intentionally follow the corrected upstream algorithm rather than retaining incorrect legacy overlap. Tests check 18,688 axis plans and 1,800 synthetic image/video round trips, scales 1/2/8, edge/oversized/irregular tile dimensions, all wrap combinations, and failure stop. The former 8-wide / 6x7 tile / circular-X assertion is now also exercised through the actual VAE retry loop. Model-level visual quality and GPU execution remain unverified.

Independent review follow-up: expand the checked-in bounded-tiling regression to an explicit image/video × encode/decode Cartesian loop (3,600 cases, 900 per combination). This removes the previous parity correlation that covered image encode and video decode only. Production tiling code is unchanged.

### Auto LoRA policy for quantized type overrides

Adapted leejet/stable-diffusion.cpp #2162 (`a068716faba635496f1a4efe112df073438ede2c`): AUTO LoRA application must account for pending tensor type overrides, not only source-file dtype statistics. ModelLoader::set_wtype_override records expected_type before conversion while get_wtype_stat still reports the source type; a float checkpoint requested as quantized therefore previously selected immediate merging. The auto-policy helper now disables immediate merging for quantized expected types, preserving the fork's existing INT8/source-quantized, stream_layers/offloaded and row-split restrictions. Explicit IMMEDIATELY and AT_RUNTIME branches are unchanged. Metadata-only CPU tests exercise the real loader's global and regex type rules, unmatched rules, float/sentinel targets, mixed models and the existing constraint combinations. No model-weight conversion or rendered LoRA image quality is claimed.

### Non-recursive safetensors index shards

Adapted leejet/stable-diffusion.cpp #2058 (`4c3cf7543da480843dacea19d82664c64cbad73a`): load index-listed shards directly with the safetensors reader instead of redispatching through the generic model loader. This rejects self-referential, cyclic and nested indexes rather than recursing, and rejects wrong-format shards while preserving normal shard registration, normalized-path deduplication, prefix handling and mmap/non-mmap tensor reads. The fork has no upstream parsed-dependency FileStamp layer, so no unrelated metadata-cache architecture was imported. Tests use tiny generated scalar shards and indexes, including a real safetensors payload with a .json suffix to verify format-specific dispatch. Existing partial-state-on-failure semantics are unchanged.

### RGBA ESRGAN alpha preservation

Adapted leejet/stable-diffusion.cpp #2029 (`c92d73c408515c94beef32161bb5960764fde7a0`): validate 4D RGB/RGBA input, pass RGB only to the existing tiled/untiled ESRGAN path, bilinear-resize alpha independently and recombine it after successful model output. Preserve the fork's compute-buffer cleanup, tile policy and failure returns. The same upscale_tensor callsite serves model upscaling and hires fix. Internal helper tests cover RGB/RGBA, two batch sizes, scales 1/2/4, tiled/untiled synthetic computation, explicit bilinear expected alpha, invalid shapes and empty-result propagation. No real ESRGAN weights or learned-image quality validation is claimed.

### LLM/vision metadata name and rank guards

Adapted the independently applicable configuration-detection portion of leejet/stable-diffusion.cpp #2067 (`ede32a621edb08c22daebfe0a7021ec3a7e92408`). Match exact weight suffixes for split/unsplit vision patch embeddings and require rank-two token/position embedding weights, so auxiliary weight_scale entries cannot overwrite hidden/vocabulary/vision dimensions or falsely enable split patch embeddings. Preserve prefix filtering, projector compatibility guards and caller-disabled vision. Existing projector test fixtures now carry realistic rank-two metadata; new tests cover both patch formats, auxiliary scales, non-default dimensions, malformed ranks and prefix controls. This is not the full INT8 convrot embedding/custom-CPU-operator integration from #2067; that still needs coordinated embedding, manager placement and backend tests.

### Bounded GGUF fallback metadata arrays

Adapted the metadata parsing portion of leejet/stable-diffusion.cpp #2062 (`168f7b88b938609f183856b61c3ae7e9e79f12a2`). The fallback GGUFReader now skips scalar/array values according to their declared element type rather than recursively treating array elements as key/value records. Reads/skips track remaining file bytes; count multiplication is bounded before seeking, including string-array length prefixes. Tests cover all 12 supported scalar types, empty/nonempty arrays, tensor descriptor/data-offset continuity, truncation and excessive counts. Preserve existing primary-reader selection and Comfy original-shape restoration; do not import the separate wide-tensor probing change in this slice. This is not comprehensive malformed-GGUF hardening: tensor-name/dimension allocation, dimension overflow and reader-state reuse remain separate review areas.

### Buffered skipping for short GGUF metadata

Adapted leejet/stable-diffusion.cpp #2223 (`bb6b128b2daf92e2c80919a311da6267369a4039`): preserve the stream buffer by ignoring metadata payloads up to 64 KiB; seek for larger payloads. Keep the remaining-byte/count guards from the preceding metadata-array fix. Expanded regression checks zero-length and 65,535/65,536/65,537/131,072-byte strings/numeric arrays plus 8,192 short strings, verifying subsequent tensor descriptors and data offsets. This is a functional validation of upstream's optimization, not a measured model-load performance claim.

### MiniMax-H3 MLP activation chunking

Adapted leejet/stable-diffusion.cpp #2257 (`5356982584d5b46a8f5cb0842927aff6443fe174`): bound token chunks before the expanded first MLP projection according to the backend buffer-type maximum and alignment, accounting for F32 activations and higher dimensions. Preserve the fork's existing explicit SiLU/multiply gate rather than replacing it with upstream's fused SwiGLU, and keep its Linear scale/precision/adapter paths. Concatenate chunk outputs in token order and materialize non-contiguous input slices. CPU tests inject buffer limits only during graph construction and run actual tiny MLP graphs, comparing batched/strided/singleton/tail cases against the unchanged unchunked path. This constrains expanded individual buffers, not total graph memory or an output tensor that itself exceeds the backend limit. Real MiniMax models, GPU allocation, quantized weights, training and LoRA adapter combinations remain untested.
