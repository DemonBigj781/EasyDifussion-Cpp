# 33 — Native Token Merging, Then Negative-Guidance Minimum Sigma

## Scope and order
Requested after the cache/step-count regression work: evaluate token merging
first, then negative-guidance minimum sigma. Both are now implemented as
independent opt-in settings with zero defaults. Do not mix them into the baseline comparison
or silently change an existing sampler, resolution, frame count, or step count.

## Current recommendation
Do not import `noahlatzel/ToMe` directly into this native C++ backend. Its README
describes a PyTorch ViT/DINOv3 fork and points to `dbolya/tomesd` for diffusion.
The diffusion-specific implementation is a better reference for a separately
tested native prototype, not a drop-in GGML extension. Its published memory
benefits must not be assumed to stack with the existing efficient attention.

Negative-guidance minimum sigma is implemented as a separate, opt-in
setting after the token-merging evaluation. It skips an entire negative/
unconditional model evaluation below a sigma threshold; it does not remove
individual negative-prompt words. It changes classifier-free guidance and can
change the result. No claim that it preserves quality better than token merging
has been established for this backend.

## Tasks
- [x] Audit diffusion-specific token merging and its license before any port.
  Map merge/unmerge operations to the native spatial attention blocks and
  available GGML CPU/CUDA/Vulkan operations. Preserve latent dimensions and
  keep AnimateDiff temporal correspondence intact.
- [x] Prototype and compare token merging against the unchanged native
  baseline. Test SD1.5 and SDXL separately; test still images before animation.
  Integrate only if measured benefits and reviewed image/motion quality justify
  the change. An identity/no-merge configuration must preserve baseline output.
- [x] After that decision/work, evaluate negative-guidance minimum sigma.
  Check compatibility with CFG++, edit/image-guidance models, scheduled CFG,
  ControlNet, and caches. Never represent a skipped negative prediction as a
  newly computed result. Keep zero threshold equivalent to the existing path.
- [x] If beneficial, expose the validated settings through native request
  parsing, Python schemas/adapters, both UIs, queued snapshots, metadata, and
  tests. Preserve existing output by default and report actual skipped work.

## Validation and completion
Use fixed models, seeds, prompts, dimensions, frames, steps, scheduler, and
precision. Separate warm-up from measured runs. Record full wall time, sampling
time, actual evaluations/reuse, sampled peak VRAM, host RAM, image differences,
and temporal artifacts. Test every optimization alone before any combination.
Do not mark this objective complete based only on another project's benchmark.

Implementation details, local measurements, tested backends, and quality limits
are recorded in `../native-sampling-optimizations.md`. The native implementation
was written for GGML; no PyTorch package or upstream source code was imported.
Initial integration is complete. Wider prompt/model/temporal-quality testing
remains necessary before recommending either setting as a default.

Baseline cache measurements and the step-count correction are documented in
`../../source/sdkit3-port-source/stable-diffusion.cpp/docs/animatediff.md`.

## Primary references inspected
- https://github.com/noahlatzel/ToMe
- https://github.com/dbolya/tomesd
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_samplers_cfg_denoiser.py
