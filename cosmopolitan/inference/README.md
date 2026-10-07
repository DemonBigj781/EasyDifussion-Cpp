# Native image inference

The portable application has an `image` command and a native text-to-image
Generate page. Both call the existing diffusion engine and the same shared
GGML used by llama.cpp and the native trainer. The command frontend is C;
a small C++ boundary catches exceptions from model creation and generation.
The existing engine, HTTP server, UI renderer and WebGPU implementation retain
their implementation languages. No Python process runs the model.

The [completed inference record](../../COSMOPOLITAN_INFERENCE_VALIDATION.md)
documents [run 37551341258](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37551341258),
built from clean source `6898a4e564c3363d1692f0aed1a2da0395a17a77`.
One unchanged executable passed SD 1.5 image generation on CPU and embedded
software WebGPU, plus CPU native HTTP generation and early cancellation, on
Windows and Linux. The tested images are 256×256 with four CPU steps or two
WebGPU steps; they establish pipeline completion, not visual quality or
support for arbitrary checkpoints. WebGPU HTTP inference and real browser
automation remain separate, unvalidated paths.

## Model and output

Supply a supported, complete diffusion checkpoint as an external data file.
The application executable contains the engine and software WebGPU stack;
it does not contain a multi-gigabyte image model.

[`PIN.json`](PIN.json) identifies the trained SD 1.5 checkpoint used for
validation by immutable repository revision, exact file size and SHA-256.
It is a 1,747,190,784-byte GPUStack GGUF conversion with F16 text encoder and
VAE components and a Q4_0 conversion of the diffusion component. Some
diffusion weights remain F16. Its model card declares CreativeML Open RAIL-M;
review the linked model terms when using or redistributing the weights.

The optional **host-side** fetch helper downloads and verifies that file:

```sh
python3 cosmopolitan/inference/fetch_model.py
```

The default destination is `cosmopolitan/out/inference/models/`. A downloaded
file is accepted only if both size and SHA-256 match the pin. Model downloads
and generated images are excluded from Git. Users can obtain compatible
weights independently; Python is not required to run the executable.

## Command line

On Linux, from the repository root after building:

```sh
/bin/sh cosmopolitan/out/easy-diffusion.exe image \
  --model cosmopolitan/out/inference/models/stable-diffusion-v1-5-Q4_0.gguf \
  --prompt "a photograph of a red apple on a wooden table, natural light" \
  --output apple.png \
  --backend cpu --width 512 --height 512 --steps 20 --seed 42 --threads 4
```

On Windows, with the same executable and model copied to the current folder:

```powershell
.\easy-diffusion.exe image `
  --model .\stable-diffusion-v1-5-Q4_0.gguf `
  --prompt "a photograph of a red apple on a wooden table, natural light" `
  --output .\apple.png `
  --backend cpu --width 512 --height 512 --steps 20 --seed 42 --threads 4
```

Use `--backend webgpu` to select the bundled software Vulkan/WebGPU path.
This runs on the CPU through lavapipe and LLVM. Unsupported GGML operations
may use the ordinary CPU backend. It is not hardware GPU acceleration, and
the program does not claim that every operation ran through WebGPU.

Model, prompt and output arguments are required. Defaults are CPU, 512×512,
20 steps, seed 42, guidance 7, and the loaded model's default sampler and
scheduler. Optional arguments include `--negative-prompt`, `--cfg-scale`,
`--sampler`, `--scheduler` and `--vae`. `image --help` lists accepted bounds.
Flash attention, SageAttention and previews are disabled in this command.
It generates one image and writes a PNG using the existing linked encoder.
Output publication uses a temporary file beside the destination followed by
rename, and the output cannot alias the model or optional VAE file.

Memory consumption includes weights, activations, attention buffers and the
software driver's allocations. Checkpoint file size alone does not determine
the RAM required by a chosen resolution. Software WebGPU and direct GGML CPU
execution have different performance and memory costs.

## Generate page

Start the native server with a checkpoint directory:

```sh
/bin/sh cosmopolitan/out/easy-diffusion.exe sdkit \
  --backend cpu --ckpt-dir cosmopolitan/out/inference/models --port 8188
```

Open `http://127.0.0.1:8188/`, select an indexed complete checkpoint and the
compute backend, enter a prompt, and choose Generate. The page displays
sampling progress, followed by the returned PNG and its download link.
CPU is the default; embedded software WebGPU is selectable when registered.

This form supports one text-to-image request at a time. Separate companion
models, LoRA, ControlNet, image-to-image, generation plugins and non-PNG output
controls are disabled in this form. Other rendered application pages still
have separate backend integration work; see
[`application-api.md`](../docs/application-api.md).

Stop becomes available after the first sampling step completes. Cancellation
during checkpoint loading is not implemented. A stop request waits for native
generation to return; it does not instantly free memory in use. If the generation connection
is lost, the page keeps the request pending until progress establishes that
the server has finished. The server advertises its native single-user mode
explicitly; kiosk configuration is unsupported, and failed capability lookup
disables generation.

## Native request contract

| Route | Behavior |
| --- | --- |
| `GET /v1/sdapi/v1/cosmopolitan-capabilities` | Native protocol version, supported form features and cancellation limits |
| `GET /v1/sdapi/v1/checkpoints` | `{"models":[{"name":"checkpoint.gguf"}]}` from the native file index |
| `POST /v1/sdapi/v1/refresh-checkpoints` | Rescan the configured checkpoint directory |
| `GET /v1/sdapi/v1/backend-devices` | Registered backend devices |
| `POST /v1/sdapi/v1/txt2img` | Generate with native parameters; returns `images`, `info` and `task_id` |
| `POST /v1/internal/progress` | Poll a specific `id_task`; includes step counts, completion and interruption state |
| `POST /v1/sdapi/v1/interrupt` | Requires the active, sampling task's `id_task` |
| `GET /v1/sdapi/v1/options` | Read persistent options without exposing a temporary request overlay |
| `POST /v1/sdapi/v1/options` | Update persistent options while generation is idle |

The portable form sends a unique `force_task_id` and an atomic request overlay:

```json
{
  "force_task_id": "image-example-1",
  "prompt": "a photograph of a red apple on a wooden table, natural light",
  "negative_prompt": "",
  "width": 256,
  "height": 256,
  "steps": 4,
  "cfg_scale": 7,
  "seed": 42,
  "batch_size": 1,
  "sampler_name": "euler",
  "scheduler": "discrete",
  "backend": "cpu",
  "override_settings": {
    "sd_model_checkpoint": "stable-diffusion-v1-5-Q4_0.gguf",
    "forge_additional_modules": [],
    "live_previews_enable": false
  }
}
```

Only these request override forms are accepted. The checkpoint must be
indexed. The overlay is in memory for that generation and never saved into
`options.json`. One nonblocking server lock covers image generation, video
generation and options mutation. A conflicting operation receives HTTP 409;
progress, checkpoint listing and persistent options reads remain available.
Duplicate task IDs receive 409. Invalid overrides receive 400. Interrupting
a different, preparing or finished task receives 409. Completed requests and
error paths release the generation lock and request overlay.

Generation routes copy their request bodies into one server-owned coordinator
and leave the HTTP I/O threads free. The HTTP response remains pending until
the native operation finishes. The coordinator owns the request gate and
parameters while model loading, generation and PNG encoding run on a pthread
with an explicitly requested and measured 8 MiB stack. The pinned runtime's
default 80 KiB stack was insufficient for recursive GGML graph traversal.

The coordinator joins that inference worker, receives its exceptions and
allocation-error state, then posts the response onto the originating HTTP I/O
thread. Progress and native cancellation remain available throughout the
request. Shutdown closes admission, joins owned work before destroying server
dependencies, and releases undelivered responses after the I/O loops stop.

## Verification and evidence scope

The host verifier runs the executable, records separate stdout and stderr,
checks unchanged model/application hashes, and independently decodes the PNG
with bounded decompression and chunk CRC validation. The default 256×256,
four-step run verifies loading, denoising and image output; it is not an
image-quality benchmark. CI uses four CPU steps and two software WebGPU
steps on each OS. Two completed steps provide a sampling-only execution
interval while keeping the software-driver checks practical.

```sh
python3 cosmopolitan/tests/verify_inference.py \
  --artifact-dir cosmopolitan/out \
  --model cosmopolitan/out/inference/models/stable-diffusion-v1-5-Q4_0.gguf \
  --backend cpu --output-dir cosmopolitan/out/inference-cpu
```

Repeat with `--backend webgpu` and a fresh output directory. The helper
sets `LP_NUM_THREADS=2` for the software driver's workers, independently of
the GGML `--threads` option; `--software-threads` changes that host test limit.

The final CI run passed both image backends on both OSes. Its API gate also
passed a real four-step CPU request, 32 progress polls while that request was
active, and a separate eight-step request cancelled after sampling began.
Each host reported two actual 8,388,608-byte inference stacks with measured
guards. Raw commands, timings, PNG/log hashes and per-stage counters are in
the completed record; local quality and diagnostic runs retain separate
artifact identities.

`IMAGE_MODEL_LOAD` and `IMAGE_GENERATION` counters have separate windows.
`IMAGE_SAMPLING` records the intervals between completed sampling callbacks,
excluding the first denoising step, text encoding and final VAE decoding.
The WebGPU verifier requires positive graph, submission, dispatch, matrix
dispatch and readback counts in this sampling interval, as well as the
embedded software provider and zero native Vulkan loader opens. It does not
measure the fraction of operations using CPU fallback.

`verify_inference_api.py` separately exercises actual native HTTP generation,
progress, cancellation, model override persistence and request contention.
These model-backed tests are run sequentially. Linux image tests use an
explicit APE loader; Windows runs the identical file as a native PE. The
existing empty-filesystem Linux and clean-directory Windows runtime tests
remain separate from these model-backed image tests. The server is required
to remain alive before harness cleanup. Linux exits 0 after SIGTERM; Windows
cleanup exits 1 because CPython calls `TerminateProcess(handle, 1)`. That
Windows result is forced test cleanup, not an inference failure or proof of
graceful signal shutdown.

## In-place storage across CPU and WebGPU

Diffusion graphs use in-place operations after group normalization. In this
backend, normalization may require CPU execution while subsequent operations
and their weights support WebGPU. An in-place tensor still writes into its
original owner's buffer; changing an input pointer to a device copy does not
move that destination. Independent placement of those operations previously
allowed WebGPU to interpret CPU tensor bytes as a WebGPU buffer object.

The shared GGML scheduler now resolves each group containing in-place compute
before creating copies between backends. Its owner, producing operation and
aliases must use a backend that supports all group operations and any existing
storage. It preserves preallocated buffers, gives storage correctness priority
over conflicting placement hints, and retains the caller's CPU fallback
policy. A later out-of-place operation can return to WebGPU through an ordinary
copy. Impossible placements fail with a named diagnostic.

`webgpu-inplace-test`, also included in `--self-test`, passed six small graph
cases and twelve independent scalar readbacks on both final CI hosts:
repeated scheduler-allocated and preallocated CPU owners followed by a WebGPU
matrix multiply, and preallocated WebGPU owners with CPU fallback disabled.
It checks actual buffers and
assignments as well as numerical results and dispatch/readback counts. The
original tests requiring entirely WebGPU graph execution remain separate.

These inference results do not validate an end-to-end training recipe,
additional model families, physical GPU execution, LibTorch/ONNX-dependent
tools or full application parity. The partial native trainer retains its
documented restrictions and strict rejection of unsupported no-fallback graphs.
