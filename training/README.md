# Training

The Training tab runs native C++ SD1.5 LoRA jobs on Intel SYCL and routes
SDXL/Anima LoRA and SD1.5 textual-inversion jobs through the Python controller.
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
  --target sdkit-sd15-trainer --jobs 10 --deploy training/native/sd15-sycl
```

The build selects sequential oneMKL threading to avoid the broken TBB package
configuration on this host. The deployed directory includes the trainer and its
GGML/stable-diffusion shared libraries. The Training service chooses this route
for SD1.5 LoRA jobs and sets the tested SYCL runtime environment.

A temporary shell launcher is available at `/tmp/train-sd15-native.sh`. It uses
`training/datasets/cat-sd15` by default, the SD1.5 checkpoint configured in the
script, 10 CPU threads, rank 16, learning rate `1e-4`, and 512px resolution.
`EPOCHS=30` means 30 full shuffled passes; the current 16-image prepared dataset
therefore runs 480 optimizer steps at batch size 1. Override `EPOCHS`, `DATASET`,
`MODEL`, `TRIGGER`, `OUTPUT_NAME`, or other values as shell environment variables.
Set `PRINT_CONFIG=1` to inspect settings without starting training. The launcher
writes weights to `training/outputs/` by default; the Training tab is the route
that publishes finished weights into the configured LoRA folder automatically.

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
four sd-scripts parsers; it does not train a model.
The browser test accepts `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` when using an
existing Playwright installation elsewhere on disk.

Packaging follows [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
and [PyInstaller spec files](https://pyinstaller.org/en/stable/spec-files.html).
