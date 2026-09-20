# Training

The Training tab runs local SD1.5/SDXL LoRA and textual-inversion jobs through
one `sdkit-trainer` controller. It supports batch WD14 captions, caption review,
progress/logs, cancellation, saved job history, and interrupted LoRA resume.
Outputs are safetensors files in the configured LoRA/embedding directory under
`trained/`. Each filename has a job suffix so previous outputs are preserved.

## Setup

From the Easy Diffusion repository root, with `uv` installed:

```sh
python3 training/trainer.py setup
python3 training/trainer.py build
training/dist/sdkit-trainer/sdkit-trainer --root "$PWD" probe
```

`setup` installs the locked Linux x86-64, Python 3.11, PyTorch 2.6/CUDA 12.4
runtime into `training/runtime/.venv`. It needs several GB of disk space and an
NVIDIA GPU compatible with that build. Other accelerators and RTX 50-series
GPUs need a separately validated runtime; set `ED_TRAINER_PYTHON` to its Python
executable. Easy Diffusion's existing Python environment is not modified.

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
and the four training scripts. See the backend's `LICENSE.md` before distributing
its source or dependencies.

## Use

1. Put training images below `training/datasets`, e.g.
   `training/datasets/my-character/photo.png`. Subfolders are supported.
2. Open Training and enter `my-character`. Scan the dataset. Autotag uses the
   existing local `/tag` endpoint and installed WD14 ONNX/CSV assets. It creates
   missing `.txt` captions; existing files, including empty captions, are skipped.
3. Scan again to review generated captions, then edit and save individual captions.
4. Choose a full SD1.5 or SDXL safetensors checkpoint, matching model family,
   output name and settings. GGUF, standalone diffusion-only weights, and other
   architectures are not training inputs for this version.
5. Finish generation and release inference VRAM before starting. Generation
   submissions are blocked during training/autotag jobs. Automatic inference-model
   unloading is not implemented; other GPU applications may also occupy memory.
6. Start training. Captions/images are copied into the job directory; the trigger
   is added to this copy. The final artifact is exported automatically. Refresh
   model selectors to use it.

LoRA trains U-Net adapters with AdamW, SDPA, latent caching and gradient
checkpointing. Embeddings train the text token and require a unique single-word
trigger plus an initialization word. Start with batch size 1. SDXL needs more
VRAM than SD1.5; the UI defaults are starting values, not tuned presets.

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
