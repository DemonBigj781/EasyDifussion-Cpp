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
revision in `source/llama.cpp/SOURCE_SNAPSHOT.md`. Prepare the native server
with `./install.sh --llama-build`; prepare the dedicated Python conversion
environment with `./install.sh --gguf-tools`. Running `./install.sh` does both.

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
