# Native spatial token merging and minimum-sigma guidance

Both settings are available in the legacy Options panel and the C++ Render
Settings panel. They apply to SD1.5/SDXL image generation and AnimateDiff.
Both default to **0**, preserving the existing sampling path. They are separate
from the Disabled/EasyCache/TeaCache selector and do not combine those caches.

## Spatial token merging

`tome_ratio` is a fraction from 0 to 0.5. This implementation is a native,
deterministic, windowed ToMe-style adaptation, not the PyTorch ViT/DINOv3 fork
and not an exact reproduction of the diffusion-specific Python package.

It partitions each frame's spatial tokens into windows of at most 1024 tokens,
uses one token out of each four as a destination, matches source features by
cosine similarity, and averages selected sources into their destinations.
Self-attention runs on the reduced sequence. Unmerging restores the original
spatial layout before the residual addition, cross-attention, feed-forward
layers, and AnimateDiff temporal processing. Frames are never merged together.
Only the two largest latent spatial scales are eligible, and sequences shorter
than 64 tokens are left unchanged. Training graphs are unchanged.

Matching and reconstruction use existing GGML operations without downloading
hidden states to Python. CUDA/CPU use matrix-based accumulation; Vulkan uses
F32 scatter accumulation to avoid reduced-precision feature averaging. The
matching window also keeps argsort within the non-CUB CUDA implementation's
supported range. Zero bypasses all merge/unmerge work.

This is lossy. Outputs can change, and lower-resolution or larger-UNet workloads
may see no benefit. ControlNet-LLLite projection patches are rejected with token
merging because their token layout assumptions differ. Other model families are
rejected rather than silently treated as SD1.5/SDXL.

## Negative guidance minimum sigma

`negative_guidance_min_sigma` is a finite non-negative value. Zero disables it.
For standard CFG, an eligible positive step with an even step number and sigma
strictly below the threshold omits the entire negative/unconditional model
evaluation. The positive prediction is used directly for that evaluation;
there is no cached or fabricated negative prediction. The first evaluation and
sampler correction evaluations retain their normal behavior.

This changes guidance, not individual words in a negative prompt. Large values
can substantially weaken negative guidance. It has no saving when the request
already does not require a negative prediction. CFG++, image-guidance/edit
conditioning, APG, SLG, and scheduled-CFG combinations are rejected when active.
Native logs report the actual number of negative evaluations omitted.
NGMS currently requires sample caches to be disabled: omitting negative passes
would break caches that assume both conditions share a refresh history. The UI
and API reject this combination instead of silently disabling either setting.

## Requests, saving, and progress

Both fields pass through image/video request schemas, Python native adapters,
native JSON parsing, and the shared sampler's existing extra-argument channel.
They are saved with UI preferences and request metadata. Native API clients use
the same `tome_ratio` and `negative_guidance_min_sigma` keys.

Video and image progress preserve the native effective step total rather than
substituting the requested total. Video status distinguishes sampling from
decoding and finalizing. GIF download filenames follow the actual data URI MIME
type even when the still-image output selector is PNG. Uniform-gray result
frames generate a diagnostic warning; no content is silently replaced.

## Measurements and validation (2026-10-25)

RTX 3060, CUDA, flash attention, fixed seed 42, CFG 7, Euler/discrete schedule,
eight requested steps. Each number is the average of two warm requests after
one excluded warm-up. ToMe ratio was 0.5; minimum sigma was 1. Tests ran each
optimization separately, with caches disabled.

| Workload | Off | Token merging | Minimum sigma |
|---|---:|---:|---:|
| SD1.5, 1024x1024 image | 18.27 s | 13.39 s | 16.90 s |
| SDXL, 512x512 image | 3.58 s | 3.61 s | 3.31 s |
| SDXL, 1024x1024 image | 14.89 s | 14.89 s | 13.94 s |
| SD1.5 AnimateDiff image-to-video, 256x256, four frames, strength 0.75 (seven effective steps) | 7.55 s | 7.59 s | 7.07 s |

These are diagnostic samples, not universal speed or quality guarantees.
Sampled peak VRAM did not decrease: token merging added about 24 MiB in the
SD1.5 1024 test and 12 MiB in the SDXL 1024 test. Therefore it is not presented
as a VRAM-saving solution for this backend. Generated pixels differ when either
optimization is enabled; no perceptual or temporal-equivalence claim is made.

Merge/unmerge correctness was exercised on CPU, CUDA, NVIDIA Vulkan, and Intel
SG1 Vulkan. Small Intel SG1 inference round trips (off/on/off) restored
byte-identical baseline PNGs independently for both optimizations. Intel VRAM
was not sampled with the NVIDIA-only monitor. CUDA work was stopped when the
operator requested the temporary Intel switch.

Tests: `token-merge-test`, `sampling-optimizations-test`,
`scripts/test_sampling_optimizations.py`, `scripts/test_webui_progress.py`,
`scripts/test_video_progress_download.cjs`, and both UI browser regressions.

The operator's gray image-to-video output has not been reproduced with the
available test image, including a 32-frame 512x576 run. Its original input/output
artifact is still needed to resolve that separate report. Fixing its filename
does not establish that the gray pixels are fixed.

## References

- https://github.com/dbolya/tomesd (diffusion-specific reference, MIT license)
- https://github.com/noahlatzel/ToMe (ViT/DINOv3 fork, not imported)
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_samplers_cfg_denoiser.py (minimum-sigma guidance behavior)
