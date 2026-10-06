# Training

Both Training interfaces expose a **Training backend** selector. SD1.5 defaults
to native C++ LoRA on Intel SYCL; select **Legacy Python (sd-scripts)** to use
the existing Python trainer, including SD1.5 embeddings. SDXL/Anima use Python.
The API field is `backend` (`native` or `python`); omitted values preserve the
architecture-based default. Readiness and native-only restrictions follow the
selected backend, not just the model family. Both pages use the existing
searchable model picker for checkpoint selection.
It supports WD14 captions, progress/logs, cancellation, saved job history, and
resume for compatible Python LoRA jobs. Web training outputs are safetensors in
the configured LoRA/embedding directory under `trained/` with unique job suffixes.

## Setup

From the Easy Diffusion repository root, with `uv` installed:

```sh
python3 training/trainer.py setup
python3 training/trainer.py build
training/dist/sdkit-trainer/sdkit-trainer --root "$PWD" probe
```

`setup` installs the locked Linux x86-64, Python 3.11, PyTorch 2.6/CUDA 12.4
runtime into `training/runtime/.venv`. It needs several GB of disk space and an
NVIDIA GPU compatible with that build. For Intel XPU, use a separately validated
PyTorch XPU environment and set `ED_TRAINER_PYTHON` to its Python executable;
the bundled CUDA environment does not install Intel packages. Easy Diffusion's
existing Python environment is not modified.

`build` uses a separate small uv environment and PyInstaller's directory bundle.
Distribute the entire `training/dist/sdkit-trainer/` directory, including
`_internal/`. Torch is loaded by the runtime subprocess. The server prefers the
built executable and falls back to `training/trainer.py` during development.

The backend defaults to `scripts/sd-scripts` at commit
`4e624302e0088e39933b31cbc71f24212e900f5f`. Set `ED_TRAINER_BACKEND` to an existing
checkout, or create an ignored `training/local.json`:

```json
{"backend_dir": "/absolute/path/to/scripts/sd-scripts"}
```

The controller checks that revision and never clones, updates, or rewrites the
checkout. The local requirements are recorded in `runtime/pyproject.toml` and
`runtime/uv.lock`. Updating the backend requires revalidating those requirements
and the five training scripts. See the backend's `LICENSE.md` before distributing
its source or dependencies.

## SpriteGPT runtime

The Training tab can start the separate SpriteGPT Studio runtime from the provided
`/mnt/38FEF88DFEF84522/sprite-diffusion` checkout. It uses that checkout’s
`spritegpt_webui.py` and `sprite-gpt-text64.pt`; this is a Python RGB pixel-space
trainer at 64×64 and does not use a VAE. Set `SPRITEGPT_ROOT` and
`SPRITEGPT_PYTHON` in the Easy Diffusion server environment to override the
checkout or interpreter. Runtime logs go to `training/logs/spritegpt-runtime.log`.

### Native C++ SpriteGPT training

The isolated `sdkit-sprite-gpt` sidecar has a LibTorch rectified-flow training
loop for the 64×64 RGB model. Build it with `source/Build.sh --sprite-gpt`
using an environment whose Python package provides LibTorch. The data folder
must contain supported images (`png`, `jpg`, `jpeg`, `bmp`, or `tga`) with a
same-basename `.txt` caption beside each image. For XPU, first export the
TorchScript U-Net and CLIP text encoder with the Intel XPU PyTorch environment
(`export_sprite_gpt.py --device xpu`); exported TorchScript constants are
device-specific. Then run:

```sh
bin/sdkit-sprite-gpt --train \
  --unet /path/to/sprite-gpt-unet.ts \
  --clip /path/to/clip-text.ts \
  --tokenizer /path/to/tokenizer \
  --dataset training/datasets/my-character \
  --trigger mycharacter \
  --output models/sprite-gpt/my-character.ts \
  --device xpu --steps 10000 --batch-size 1
```

Training saves an updated TorchScript U-Net and a matching `.safetensors`
parameter export at periodic intervals and at completion. This native command
is currently a CLI path; the SpriteGPT Studio button still starts the existing
Python runtime. The C++ target compiles, but this host's LibTorch package is
CUDA-only, so an Intel XPU training run has not yet been validated here.

For scraped datasets, create a folder in the Training tab, set [Grabber](https://github.com/Bionus/imgbrd-grabber)
to download images there, then add/review same-basename `.txt` captions.

## Grabber scrape gallery

The Training tab queries the installed Grabber CLI and groups each search into a
separate attempt gallery. It provides before/filtered views, select-all download,
review/delete/tag actions, and archive after captions are present. Still images,
GIF samples, and video frames are normalized to PNG at no more than 2048×2048 and
use the MD5 of the resulting PNG plus resolution in the filename. Animated media
is sampled at intervals no greater than 10 seconds or 600 frames. The near-duplicate
cutoff defaults to 0.95 and is configurable per attempt. Gallery autotagging saves
original `.txt` captions beside the images before replacing them. Set `GRABBER_CLI`
if Grabber's CLI is not at `/home/jack/bin/grabber/Grabber-cli`.
The website picker reads Grabber's installed source list and hides entries listed in `training/.grabber-source-blacklist` (one exact host per line). ArtStation is hidden by default because its adapter is no longer reliable for image searches.

## Native C++ SD1.5 LoRA

Build and deploy the SYCL trainer from the repository root:

```sh
source /opt/intel/oneapi/setvars.sh --force
source/Build.sh --sycl --build-dir /tmp/easy-diffusion-native-sycl \
  --target sdkit-sd15-trainer --jobs 1 --deploy training/native/sd15-sycl
```

The build selects sequential oneMKL threading to avoid the broken TBB package
configuration on this host. The deployed directory includes the trainer and its
GGML/stable-diffusion shared libraries. The Training service chooses this route
for SD1.5 LoRA jobs and sets the tested SYCL runtime environment.

The Training UI defaults to 1200 steps, a UNet learning rate of `5e-5`, and an
SD1.5 text-encoder learning rate of `5e-6`. These are initial settings, not a
guarantee of image quality; inspect intermediate checkpoints before a full run.
The C++ page saves edited settings in the browser, and never starts a job on load.
Its settings are grouped beside a scrollable jobs panel on desktop and stacked
on narrow screens. Historical logs wrap inside a bounded panel rather than
widening the page. Anima-only fields are hidden for other families; SpriteGPT
is a separate collapsed runtime section. Native precision is fixed, not a
selectable FP32 mode.

For native CLI jobs, use `--learning-rate 0.00005 --text-encoder-learning-rate 0.000005 --steps 1200`
alongside the required model, dataset and output arguments. The API field is
`text_encoder_learning_rate`. Its API/CLI default is `0`, preserving the previous
frozen-text-encoder behavior for existing clients; positive values are supported
for SD1.5 LoRA jobs on either backend. The Python path passes separate
`--unet_lr` and `--text_encoder_lr` values and does not force UNet-only training
when the text-encoder rate is positive. Unsupported families reject positive values
rather than silently ignoring them. The old temporary `/tmp` launcher is not
required or shipped; use the Training page or the native executable directly.

With a positive text-encoder rate, CLIP runs in the same epsilon-prediction loss
graph as UNet, with its own AdamW rate. Only VAE latents and caption token IDs are
cached; CLIP hidden states are recomputed every step. Captions are literal text,
not weighted inference-prompt markup. The base model remains frozen; the saved
LoRA includes 96 CLIP attention factors plus 256 UNet attention factors at rank
16 (other ranks change sizes, not tensor counts). Setting the text-encoder rate
to `0` omits CLIP factors and retains cached conditioning. Both paths export
normal safetensors LoRAs. Checkpoint resume remains unsupported for native jobs.

### Captions, epochs and samples

Leave **Optional extra tag** blank to condition on each image's caption without
adding a placeholder token. Caption words are conditioning text, not guaranteed
independent activation tokens. The native graph uses a 77-token CLIP sequence;
keep captions within 75 content tokens plus start/end tokens to avoid truncation.

For native training, `dataset_repeats` defaults to 1. With batch size 1, an epoch
contains `image_count × dataset_repeats` optimizer steps. Optional `epochs`
overrides the step count; otherwise `steps` remains the stopping limit. The CLI
equivalents are `--dataset-repeats` and `--epochs`. The final partial epoch does
not emit an epoch-end sample. Each completed epoch saves a checkpoint, emits
`epoch_end`, then synchronously samples before the next update. A sampling error
fails the job rather than silently continuing without a sample.

Native samples use the first staged caption, an empty negative prompt, the job
seed, 25 deterministic DDIM steps and guidance 7.5. They are saved beneath the
job's `output/samples/`, with a matching caption file; `sample_start` records
settings and `sample_saved` records the path. Sampling uses the current UNet and
CLIP factors through the forward-only optimizer graph, not a second inference
service or a fresh training process. Python jobs use sd-scripts'
`sample_every_n_epochs=1` and a matching `sample-prompts.json`; their images are
under `output/sample/`. Native and Python samplers are not claimed pixel-identical.

The export uses interoperable `lora_unet_…` / `lora_te_…` module names with
`.lora_down.weight`, `.lora_up.weight` and scalar `.alpha` entries. Verification
with the installed ComfyUI loader checks target shapes and scale for every
UNet/CLIP pair; native name conversion is tested independently. Existing
checkpoints are not rewritten automatically. Changing names cannot undo training
with an unwanted caption token.

Epoch scheduling, forward-only sampling without factor updates, export and
controller paths have automated coverage. End-to-end GPU epoch sampling and
sample quality still require an explicitly authorized training run.

### Alpha and learning-rate scheduling

`network_alpha` is independent of `rank`. Omit it (or send `null`) to use the
rank, preserving existing clients. A supplied alpha must be finite, greater
than zero and at most 128. Both Training interfaces expose this as **LoRA alpha**;
leaving the field blank follows the current rank. Native export writes a scalar
`.alpha` for every UNet/CLIP factor pair and records `ss_network_dim` and
`ss_network_alpha` metadata. Inference therefore uses the same `alpha / rank`
scale as training, including rank 32 with alpha 16. Factors remain F32.

`lr_scheduler` accepts `constant` (default) or `cosine_with_restarts`.
`lr_warmup_steps` defaults to zero; `lr_scheduler_num_cycles` defaults to one.
These are integer counts: warmup must be less than total training steps, and
cycles must be positive and no greater than the number of post-warmup steps.
Constant scheduling requires zero warmup and one cycle. One cosine cycle has
no intermediate restart; two cycles have one restart. Disabled UI controls
submit the constant defaults without destroying the user's saved cosine values.

The cosine schedule uses the zero-based optimizer-update index, matching the
Hugging Face hard-restart schedule. With warmup enabled, the first update has
zero learning rate: moments advance, but parameter values do not change.
After warmup, each cycle decays from the base rate toward zero and restarts at
the base rate. Both UNet and CLIP receive the same multiplier, preserving their
configured rate ratio. JSONL progress reports the rates actually used for that
update. No optimizer-state reset occurs at a restart.

Native CLI options are `--rank 32 --network-alpha 16`,
`--lr-scheduler cosine_with_restarts`, `--lr-warmup-steps 0`, and
`--lr-scheduler-num-cycles 1`.
The controller also forwards these settings to the configured Python LoRA
backend rather than silently replacing them with rank-equal alpha or constant
scheduling.

Native AdamW8bit and selectable FP16 training/export remain unimplemented.
The native service rejects AdamW8bit instead of substituting AdamW. This change
does not make the prepared exact 1500-step recipe runnable, repair old
checkpoints, or establish fresh-training image quality.
Native weight-decay and epsilon overrides also remain unexposed by the CLI;
the prepared recipe's optimizer parameters must not be silently substituted.

Captions longer than the SD1.5 CLIP context are truncated to 77 tokens during
native training. Existing captions should be reviewed before a full run. The
prepared `cat-sd15` folder is a separate copy of `/mnt/1812FB8512FB6662/cat`; its
source images were left unchanged, and missing captions were filled by the C++
WD14 ONNX tagger at threshold 0.35. WD14 can mislabel features, so edit its `.txt`
files before relying on those labels.

## Use

1. Put training images below `training/datasets`, e.g.
   `training/datasets/my-character/photo.png`. Subfolders are supported.
2. Open Training and enter `my-character`. Scan the dataset. Autotag uses the
   existing local `/tag` endpoint and installed WD14 ONNX/CSV assets. It creates
   missing `.txt` captions; existing files, including empty captions, are skipped.
3. Scan again to review generated captions, then edit and save individual captions.
4. Choose a full SD1.5, SDXL, or Anima safetensors checkpoint and matching model
   family. Anima additionally needs Qwen3 and Qwen-Image VAE paths inside the
   configured text-encoder and VAE directories. GGUF and other architectures
   are not training inputs for this version.
5. Finish generation and release inference VRAM before starting. Generation
   submissions are blocked during training/autotag jobs. Automatic inference-model
   unloading is not implemented; other GPU applications may also occupy memory.
6. Start training. Captions/images are copied into the job directory; the trigger
   is added to this copy. The final artifact is exported automatically. Refresh
   model selectors to use it.

The dataset editor can create folders under `training/datasets`. Point Grabber's
download target there and provide same-basename `.txt` tag sidecars for each
image. Anima captions convert underscores to spaces except for tags beginning
with `@`, which retain artist-name underscores. Anima jobs use `networks.lora_anima`,
latent caching, SDPA, the Qwen-Image 2D VAE path, and VAE chunking. The UI offers
standard, CPU-offloaded, or Unsloth checkpointing, block swapping, text-output
caching, and AdamW/AdamW8bit. Unsloth and 8-bit optimizers require their runtime
dependencies. Start with batch size 1; settings are not tuned presets.
SDXL and Anima continue to use the configured Python runtime and Kohya-based
scripts. Native SD1.5 LoRA training currently requires batch size 1, a resolution
from 256 through 1024 in multiples of 64, and targets SYCL device `SYCL1`. It
uses the C++ trainer and does not yet support resume, embeddings, gradient
checkpointing, or automatic memory offload.

Jobs live in `bucket/training/<job-id>/`: `job.json`, `manifest.json`,
`events.jsonl`, dataset snapshot/config, output weights and Accelerate state.
These local files include captions and paths. No Hub upload occurs. Tokenizer
files may be downloaded by sd-scripts on first use.

Cancel stops the trainer process tree; it does not create an extra checkpoint.
Autotag cancellation waits for its current HTTP request (up to 60 seconds).
Resume is available for failed/cancelled/interrupted **LoRA** jobs after a saved
state exists, using that job's dataset snapshot. Embedding resume is not exposed:
the upstream script does not restore the same global-step behavior. A server
restart marks unfinished jobs interrupted. Saved state and snapshots consume
disk until you remove a finished job directory yourself.

## API and protocol

All web paths are under `/training`: `GET /readiness`, `GET /models`,
`POST /dataset/scan`, `POST /dataset/caption`, `POST /autotag`, `POST /jobs`,
`GET /jobs`, `GET /jobs/{id}`, and `POST /jobs/{id}/cancel` or `/resume`.
Datasets must be relative to `training/datasets`; checkpoint paths must be inside
configured checkpoint roots. Executable paths, output directories and arbitrary
backend arguments cannot be supplied through the web API.

The controller accepts one JSON line on stdin with `command` equal to
`autotag`, `train-lora`, `train-embedding`, or `resume`, and validated job fields.
It emits JSONL `started`, `dataset`, `log`, `progress` and terminal
`completed`/`failed`/`cancelled` events. Keep stdin open for the lifetime of the
job; send `{"command":"cancel"}` to cancel. Closing stdin also cancels the job.
The server controls this protocol; most users only need the Training tab.

## Validation

```sh
uv run --project training --locked --group test python scripts/test_training.py -v
training/runtime/.venv/bin/python scripts/test_training_backend.py
node --check ui/plugins/ui/training_plugin/training.tab.plugin.js
# Optional browser component test, with Playwright and Chromium installed:
node scripts/test_training_ui.cjs
```

The first suite covers filesystem boundaries, caption preservation, CLI/bundled
autotagging, subprocess cancellation, job persistence, export, and HTTP errors.
The second checks generated arguments and dataset configs against the actual
five sd-scripts parsers; it does not train a model.
The browser test accepts `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` when using an
existing Playwright installation elsewhere on disk.

Text-encoder rate validation is covered by `python -m unittest scripts.test_text_encoder_rate`.
The C++ page has a separate browser test: `node scripts/test_cpp_training_ui.cjs`.
Set `TRAINING_TEST_URL` to its `/cpp-ui/training` URL. All training API calls in
that test are intercepted, so it never starts a job. With `SDKIT_BUILD_TESTS=ON`,
the CMake target `text-encoder-optimizer-test` checks actual AdamW update ratios
on CPU. `scripts/verify_sd15_text_encoder_checkpoint.py` verifies the joint and
frozen two-step smoke artifacts: finite values, changed CLIP/UNet factors, and
matching initial losses before adapter updates.

The `sd15-backward-math-test` target compares GELU, QuickGELU and SiLU gradients
with independent scalar derivatives. CTest runs the CPU cases. Pass a backend
name, such as `sd15-backward-math-test SYCL2`, to check that accelerator as well.
This regression covers backward-graph constants: host-only scalar leaves were
not reliably materialized on SYCL, producing incorrect GELU/QuickGELU gradients
despite finite losses and weights. Constants are now computed by graph
operations. Checkpoints trained with the affected implementation need fresh
training; loading them with the corrected library does not repair their weights.

The `optimizer-compute-failure` CTest runs `optimizer-failure-fixture` both
normally and with a CPU backend abort callback. Normal execution must succeed;
an aborted computation must terminate with `ggml_opt: graph computation failed`
rather than return an apparently valid loss. The optimizer also checks graph
allocation failures before marking a graph ready. These checks fail fast by
aborting the trainer process; they do not roll back partially executed updates
or implement checkpoint recovery. The fixture exercises compute failure, not
allocation failure.

For numerical investigations, capture raw gradients before any optimizer step,
check the mean-squared-error loss seed and fresh zero-up LoRA invariant, and
compare against forward-only finite differences with fixed inputs. Verify that
all parameters are restored exactly, repeat the unperturbed loss to measure
noise, and retain unresolved or disagreeing results rather than treating a
completed diagnostic as a pass. Full-F32 diagnostic graphs are a separate
reference path, not proof of mixed-precision or full-model CPU parity. Passing
these checks does not establish fresh-training image quality.

Packaging follows [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
and [PyInstaller spec files](https://pyinstaller.org/en/stable/spec-files.html).
