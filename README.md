# Easy Diffusion Custom

This local Easy Diffusion customization project adds native sdkit3 image-generation features
and local model tooling:

- Native ControlNet-LLLite conditioning for the sdkit3 backend.
- Automatic VAE-to-model encode and model-to-VAE decode latent conversion
  using the city96 v4.0 interposer matrix.
- WD14 image tagging through the built-in Easy Diffusion `/tag` endpoint.
- Built-in same-origin file discovery under `/files/*` and LoRA metadata APIs
  under `/meta/*`, plus modern Base64 bucket compatibility.
- Built-in Perchance image, text, and public-gallery APIs under `/perchance/*`,
  including persistent gallery image/URL ID and generator channel settings.
- An Online Model Browser for Civitai and Hugging Face, using separate
  `/civitai-api` and `/huggingface-api` routes.
- A de-gitted llama.cpp source snapshot, a CUDA-capable local `llama-server`
  backend for TIPO, and Hugging Face model-to-GGUF conversion in Model Tools.
- Native GIF output/GIF-to-GIF tasks and an opt-in manager for the supplied
  legacy UI plugins, including the repaired Rabbit Hole and Storyteller tabs.
- Shell-free native-backend argument editing and idle-time backend reload from
  System Settings.

Each settings tab is an independent UI plugin: `controlnet.plugin.js`,
`controlnet-preprocessor.plugin.js`, `controlnet-lllite.plugin.js`,
`latent-interposer-encode.plugin.js`, `latent-interposer-decode.plugin.js`, and
`wd14-tagger.plugin.js`. Initial and Reference Images are owned together by the
core `reference-images.plugin.js`; their former built-in HTML was removed.
The core `perchance.plugin.js` exposes image, text, gallery list/get, download,
cursor, filtering, sort, time-range, visible-browser, and saved ID/channel
controls. Perchance uses `/home/jack/bin/perchance` and does not open a helper
HTTP port.

Model directories in the user's shared model store (also reachable through
Easy Diffusion's `models` symlink):

- `controlnetLLLite/` — kohya-ss LLLite `.safetensors` files.
- `interpose/` — city96 v4.0 converters for all available non-Cascade
  v1/SDXL/SD3/Flux source routes.
- `vae/Furception/` — `furception_vae_1-0.safetensors`, kept separate from
  converter weights.
- `deepdanbooru/` — matching WD14 `<name>.onnx` and `<name>.csv` files.
- `controlnet`, `ControlNet`, and `ControlNetPreprocessor` retain the user's
  existing native/WebUI ControlNet and preprocessor assets.
- `Controlnet_Union/`, `Uni_Controlnet/`, and `Controlnet_LITE/` keep the
  architecture-specific native ControlNet checkpoints out of the ordinary
  ControlNet picker. Their dedicated UI panels read these folders directly.
- `Image_GGUF/` receives GGUF files created by Model Tools, separate from
  language-model GGUF files used by TIPO.

Both native generation features require the locally patched sdkit3 build.
WD14 runs in Easy Diffusion's Python process and is backend-independent.

The de-gitted source snapshot for that native backend is tracked in
`source/sdkit3-port-source`. Its upstream revisions and snapshot exclusions are
recorded in `source/sdkit3-port-source/SOURCE_SNAPSHOT.md`.

The llama.cpp snapshot is tracked in `source/llama.cpp`, with its pinned
revision in `source/llama.cpp/SOURCE_SNAPSHOT.md`. It is integrated into the
sdkit native build as a colocated `llama-server` executable. llama.cpp and
stable-diffusion.cpp pin different ggml revisions, so the two runtimes remain
in separate processes rather than exposing colliding ggml C symbols in one
address space. TIPO discovers the bundled executable automatically.

Prepare the standalone llama tools with `./install.sh --llama-build`; install
the Python conversion dependencies into Easy Diffusion's main Python 3.13
environment with `./install.sh --gguf-tools`.
Running `./install.sh` does both. Build sdkit/stable-diffusion.cpp and its
integrated llama runtime together with:

The native bundle also carries llama.cpp's Hugging Face, tokenizer-update,
legacy GGML, and LoRA GGUF converter scripts. They execute out of process with
the main Python environment because their PyTorch/Transformers code cannot be
linked into the C++ diffusion process.

```bash
./install.sh --native-build --cuda
# or CPU-only
./install.sh --native-build --cpu
```

For an Intel oneAPI build, first install the card driver and oneAPI toolkit,
source Intel's `setvars.sh`, confirm that `sycl-ls` exposes a GPU, and run:

```bash
./install.sh --native-build --sycl
```

The H3C XG310 is four independent 8 GB Intel SG1/Xe-LP devices, not one unified
32 GB device. The build path is ready, but SG1 is not in llama.cpp's currently
documented list of verified SYCL devices; select the `sycl` backend only after
the installed H3C/Intel driver exposes the card through `sycl-ls`.

System Settings → Native backend arguments accepts advanced sdkit arguments
such as `--vae-tiles 32 --vae-tiled-overlap 16`. Arguments are stored as an
argv list and never passed through a shell. Easy Diffusion rejects injected
port, parent-process, and model-directory flags. Enable “Reload native backend
when saving” to apply them without restarting the UI; the queue must be empty.

## Optional local plugins

System Settings → Optional local plugins controls the supplied legacy plugin
set. Enabling a plugin loads it immediately; disabling takes effect on the next
page load because those older plugins have no unload lifecycle. GIF output is
enabled by default. The Rabbit Hole compatibility fixes accept both the old
Easy Diffusion 3.5 `/get/models` response and the current model response, while
its original file remains usable in the old 3.5 plugin directory.

The merged Danbooru/e621 CSV is parsed by `spell-tokenizer.worker.js` off the UI
thread, so loading roughly one million tag rows does not freeze the page. The
canonical Animate file forwards to the complete bundled copy and both copies
share a load guard. The repaired Storyteller implementation uses the current
tab API with a fallback for the old tab wiring.

Additional non-conflicting plugins from Tangent101's `patch-14` branch are
included as opt-ins. Its image-editor/modifier replacements, second queue
counter, and scrolling-pane layout were intentionally not duplicated because
this fork already has newer core equivalents or, in the latter case, it
conflicts with Rabbit Hole's layout.

Easy Diffusion runtime configuration is stored in the root `config.yaml`; a
root `options.json` is not used. The native backend keeps its private API state
inside its own backend working directory. `./developer_console.sh` opens the
contained `.venv`, validates `config.yaml`, and exposes its path to the shell.

The active conversion matrix is v1↔SDXL, v1↔SD3, SDXL↔SD3, and Flux→v1,
Flux→SDXL, or Flux→SD3. Same-family sources need no converter. Destination
Flux conversions and Stable Cascade are not exposed. Cascade is deferred until
Easy Diffusion and the selected sdkit3 model-loading path can be verified to
support it safely.

## Licensing and upstream attribution

- `kohya-ss/ControlNet-LLLite-ComfyUI` — GPL-3.0.
- `city96/SD-Latent-Interposer` — Apache-2.0.
- `pythongosssss/ComfyUI-WD14-Tagger` — MIT.
- WD14 models by SmilingWolf; model-specific terms remain applicable.

The project license is in `LICENSE`. Full license paths, source links, bundled
source notices, and attribution-only notices for plugins that arrived without
a standard license are listed in `THIRD_PARTY_NOTICES.md`.
