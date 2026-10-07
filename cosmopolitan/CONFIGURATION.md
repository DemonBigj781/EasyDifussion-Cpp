# Persistent configuration and compute selection

The portable executable reads `easy-diffusion.json` in the launcher's working directory. Its first ordinary `shell`, `infer`, `image`, `llama`, `sdkit`, or `devices` command creates this file if it is missing, using CPU, automatic provider policy, and automatic device selection. An invalid existing file stops startup with a configuration error; it is never silently replaced.

Use an explicit path when launching from different directories. From the folder containing the executable, Windows PowerShell commands are:

```powershell
.\easy-diffusion.exe config init --config .\settings\easy-diffusion.json
.\easy-diffusion.exe config show --config .\settings\easy-diffusion.json
.\easy-diffusion.exe config validate --config .\settings\easy-diffusion.json
.\easy-diffusion.exe sdkit --config .\settings\easy-diffusion.json
```

On Linux, run the same executable through its shell bootstrap:

```sh
/bin/sh ./easy-diffusion.exe config init --config ./settings/easy-diffusion.json
/bin/sh ./easy-diffusion.exe config show --config ./settings/easy-diffusion.json
/bin/sh ./easy-diffusion.exe config validate --config ./settings/easy-diffusion.json
/bin/sh ./easy-diffusion.exe sdkit --config ./settings/easy-diffusion.json
```

The Linux shell bootstrap temporarily extracts the bundled APE loader. Alternatively, replace `/bin/sh ./easy-diffusion.exe` with `./ape-x86_64.elf ./easy-diffusion.exe` using the supplied executable loader. Both methods leave the application bytes unchanged.

`config init` refuses to overwrite an existing file. `config defaults` prints the complete schema without reading or creating any file. Help, version, and the embedded numerical self-tests do not create a configuration file. `webgpu-device-test` reads an existing config, or uses defaults without creating a file. The native trainer retains its separate command-line contract; inference configuration flags do not apply to `train`.

A generated configuration is:

```json
{
  "schema": 1,
  "server": {"port": 8188, "log_level": "info"},
  "models": {"checkpoint_dir": "models/checkpoints"},
  "compute": {"backend": "cpu", "provider": "auto", "device": "auto"},
  "inference": {
    "image": {
      "prompt": "", "negative_prompt": "",
      "width": 512, "height": 512, "steps": 20,
      "cfg_scale": 7, "seed": 42,
      "sampler_name": "euler_a", "scheduler": "discrete"
    }
  },
  "options": {
    "sd_model_checkpoint": "",
    "live_previews_enable": false,
    "show_progress_every_n_steps": 5,
    "CLIP_stop_at_last_layers": -1,
    "sdxl_clip_l_skip": false,
    "samples_format": "png",
    "forge_additional_modules": []
  }
}
```

Relative checkpoint directories resolve beside the configuration file. A command-line `--ckpt-dir` resolves relative to the current working directory. Windows absolute drive paths such as `C:\models` are converted to Cosmopolitan's `/C/models` spelling internally; drive-relative paths such as `C:models` are rejected. The local server continues to bind to `127.0.0.1`; the configuration does not add a remote listen address.

Precedence is **defaults, saved file, command-line overrides**. `--config`, `--backend`, `--provider`, `--device`, `--port`, `--log-level`, and `--ckpt-dir` can override the effective settings for that invocation. They do not overwrite saved settings. An explicit `config init` may include overrides to choose initial saved values. Unknown JSON keys, duplicate JSON keys, malformed types, invalid enum values and out-of-range ports are rejected. Files are limited to 1 MiB.

The supported compute fields are:

| Field | Values | Meaning |
| --- | --- | --- |
| `backend` | `cpu`, `webgpu` | CPU GGML execution or the shared GGML WebGPU backend. |
| `provider` | `auto`, `native`, `embedded` | Automatic selection prefers physical hardware; native uses the operating system Vulkan path; embedded uses the included Mesa software driver. |
| `device` | `auto`, exact registry selector, available stable ID | A concrete WebGPU selector such as `WebGPU0` or a driver-provided stable ID; CPU accepts `auto` or `CPU`. Unknown explicit selections fail. |

## Saved inference defaults and the shared shell

`inference.image` stores the portable image recipe. Existing schema-1 files
without that section load with its defaults; inspection does not rewrite an
older file. The next successful save includes the normalized section.
Unknown fields and malformed recipe values remain errors. Changes to this
section are live and do not require restarting the application.

The Generate page reads the checkpoint from the native options store and the
recipe from this configuration. **Save generation defaults** sends one
`POST /v1/sdapi/v1/settings` request containing `options` and `inference`.
Checkpoint validation and recipe validation precede one atomic save. Either
both become live, or neither is persisted. The route returns the config
document and rejects updates with 409 while inference is admitted.

The [C shell](SHELL.md) uses the same services and file. For example, after
starting `shell --serve`, the following commands update values visible to
the UI and HTTP API:

```text
config set inference.image.steps 20
options set sd_model_checkpoint "an-indexed-checkpoint.gguf"
config show
```

`infer image` and the HTTP image operation fill missing recipe fields from
these effective defaults. Request fields and checkpoint overrides apply
only to that request. The shell's `cd` does not change the server's working
directory or the configuration-relative model directory. Provider/device
startup changes continue to require restart; a running shell never resets
the initialized backend registry to apply them silently.

## Select and verify a native GPU

Windows native operation requires an installed compatible Vulkan graphics driver providing `vulkan-1.dll`. Linux requires `libvulkan.so.1` and a compatible driver; Cosmopolitan 4.0.2 also builds a native library helper on first use, so a working host C compiler must be on `PATH`. No display server is needed. The backend requires `ShaderF16`, including for its F32 matrix kernels; detected adapters missing required features are reported with an unavailable reason.

The following sequence creates a **new** `gpu.json` with native WebGPU defaults, lists compatible devices, and requires actual hardware before running the trained text fixture. On Windows PowerShell:

```powershell
.\easy-diffusion.exe config init --config .\gpu.json --backend webgpu --provider native --device auto
.\easy-diffusion.exe devices --config .\gpu.json
.\easy-diffusion.exe webgpu-device-test --config .\gpu.json --device WebGPU0 --require-hardware
.\easy-diffusion.exe llama --config .\gpu.json --device WebGPU0 --prompt "Once upon a time" --tokens 16 --report-tokens
```

On Linux:

```sh
/bin/sh ./easy-diffusion.exe config init --config ./gpu.json --backend webgpu --provider native --device auto
/bin/sh ./easy-diffusion.exe devices --config ./gpu.json
/bin/sh ./easy-diffusion.exe webgpu-device-test --config ./gpu.json --device WebGPU0 --require-hardware
/bin/sh ./easy-diffusion.exe llama --config ./gpu.json --device WebGPU0 --prompt "Once upon a time" --tokens 16 --report-tokens
```

Replace `WebGPU0` with the desired selector printed by `devices`; it is an
example, not a promise about a particular GPU. Keep the same provider policy
when enumerating and using a selector. Use `--device auto --require-hardware`
in the diagnostic if any compatible physical GPU is acceptable. The diagnostic
runs four real tensor graphs, twelve scalar comparisons, and out-of-place and
in-place GroupNorm checks with uneven channel groups; adapter enumeration alone
is insufficient. `--require-hardware` belongs to `webgpu-device-test`, not the
`image`, `llama` or server commands.

If `gpu.json` already exists, skip `config init`. Inspect it with `config show`; edit `compute.backend`, `compute.provider` and `compute.device` or save them through Settings, then run `config validate`. The `--device` overrides above affect only those invocations. To persist an exact selection, save it in `compute.device`; provider/device startup changes require restarting a running server. For explicit bundled software operation, use `--backend webgpu --provider embedded --device auto` with `image`, `llama` or `sdkit`.

`llvmpipe`/lavapipe is software computation on the CPU. Its presence does not demonstrate a physical GPU. Native automatic selection fails if compatible physical hardware is absent; an explicitly selected native software adapter is useful for loader diagnostics but still counts as software. `--require-hardware` requires a native discrete/integrated GPU and rejects software and unknown adapter types. The available local and CI driver coverage is software-only; physical GPU execution remains unvalidated. Device enumeration or a selected device name does not establish support for every GPU model.

Registry selectors can change when hardware or provider policy changes. The UI uses a stable ID only when one is actually available. The pinned native API currently exposes no stable UUID/LUID to this wrapper, so present selections use the exact registry selector rather than an invented persistent hardware identity.

The embedded numerical `--self-test`, `webgpu-test`, and `webgpu-inplace-test` commands deliberately select the embedded provider, independent of saved settings. The distinct `webgpu-device-test` command honors configuration and explicit provider/device overrides, defaulting its effective backend to WebGPU because it tests that backend. CPU inference does not require the system Vulkan loader or a host compiler. WebGPU inference retains GGML CPU fallback for unsupported operations; selecting WebGPU does not mean every operation executes on a GPU.

## Settings, live options, and restart behavior

The existing Settings and GPU Config pages use the persisted native configuration. They show the file path, running settings, saved settings, actual adapter descriptions, and restart requirements. Unsupported CUDA/ROCm/SYCL backend choices and browser-only per-module selectors are disabled or removed from these portable pages. The Generate page can choose an available device for a request. Changing a provider policy in Settings requires restarting the server before the new provider is enumerated.

The configuration API is:

- `GET /get/app_config` or `GET /v1/sdapi/v1/config`: returns `config_path`, `saved`, `effective`, and `restart_required`.
- `POST /app_config` or `POST /v1/sdapi/v1/config`: merges validated `server`, `models`, and `compute` sections into the saved file. Startup fields take effect after restart. The legacy single `backend_platform` field is accepted as an alias for `compute.backend`.
- Existing `GET/POST /v1/sdapi/v1/options`: reads or saves the configuration's generation options immediately. Request-specific `override_settings` are temporary and never persisted.

Both mutation routes return 409 while a generation is active. Validation failures return 400. Filesystem failures return an error and leave the in-memory saved settings unchanged. Writes use a flushed temporary file followed by replacement; the server detects an external edit before saving and asks for a restart. This is a single-server configuration mechanism, not a multi-process transaction service.

The old working-directory `options.json` is no longer the portable application's settings store and is not imported automatically. If migrating selected options, copy supported values into the generated configuration and run `config validate`. Live previews, companion-module lists and non-PNG output remain unsupported and are rejected. Kiosk configuration remains explicitly unavailable; persisted application settings do not introduce or silently bypass a kiosk policy.

## Focused checks without models

The probe compiles the production configuration module with the pinned SDK into one portable executable. It does not initialize any WebGPU driver or model. Build it once on Linux after the main recipe has prepared `cosmopolitan/out/sdk`:

```sh
python3 cosmopolitan/tests/build_config_probe.py
python3 cosmopolitan/tests/verify_config.py \
  --probe cosmopolitan/out/config-test/config-probe.exe \
  --loader cosmopolitan/out/config-test/ape-x86_64.elf
node cosmopolitan/tests/verify_config_ui.js
```

Copy that same `config-probe.exe` to Windows and run:

```powershell
python cosmopolitan/tests/verify_config.py --probe config-probe.exe
```

The verifier checks first-run creation, strict validation, relative path resolution, CLI precedence without saved-file mutation, persistent live options, startup changes requiring restart, unchanged files on rejected updates, and configuration-independent diagnostics. `BUILD.json` records source hashes and the test executable hash. The Settings test uses a small DOM/fetch mock to check configuration errors, actual selector/stable-ID requests, truthful software labels, restart notices, provider changes, and busy responses. It does not claim browser rendering, HTTP server execution, hardware discovery, or model inference. The native API and same-artifact integration gates cover those separate boundaries.

A separate policy unit links the actual `wgpu_devices.cpp` against explicitly mocked Vulkan registration/provider selection and a mocked WGPU instance factory. It checks hardware-first automatic selection, embedded fallback, native automatic rejection of software/unknown adapter types, explicit native software selection, missing/ambiguous selectors, configured defaults versus the last request, immutable initialized policy, and the Linux main-thread guard. It does not claim actual Vulkan loading or GPU execution:

```sh
python3 cosmopolitan/tests/verify_device_policy.py --build
```

This consumes the prepared SDK and pinned foundation headers under `cosmopolitan/out/software-webgpu/foundation`. The resulting `cosmopolitan/out/device-policy-test/device-policy-test.exe` can be copied unchanged to Windows and checked with `python cosmopolitan/tests/verify_device_policy.py --probe device-policy-test.exe`. Build metadata records the production source and exact header hashes. This small policy test does not build LLVM, Mesa, Rust, or the full application.


The model-free real-server smoke test checks first-run file creation, both configuration GET aliases, packaged Settings/GPU scripts, validated disk persistence, unchanged running startup values before restart, live native options, rejected mutations leaving the file unchanged, and a real stop/restart that applies saved settings:

```sh
python3 cosmopolitan/tests/verify_configuration_api.py \
  --artifact-dir cosmopolitan/out --logs cosmopolitan/out/configuration-api-results
```

It uses fresh directories and a sanitized environment, limits the full check to 60 seconds by default, preserves separate stdout/stderr logs and copies of the generated JSON, and verifies the application hash before and after. It starts with the embedded provider and does not load a model. Busy-generation rejection belongs to the separate real inference API verifier. Windows cleanup uses the host's `TerminateProcess` status 1; Linux SIGTERM must exit 0. This check does not claim browser execution or physical GPU coverage.
