# Easy Diffusion Custom

This local Easy Diffusion customization project adds native sdkit3 image-generation features:

- Native ControlNet-LLLite conditioning for the sdkit3 backend.
- Automatic VAE-to-model encode and model-to-VAE decode latent conversion
  using the city96 v4.0 interposer matrix.
- WD14 image tagging through the built-in Easy Diffusion `/tag` endpoint.
- Built-in same-origin file discovery under `/files/*` and LoRA metadata APIs
  under `/meta/*`, plus modern Base64 bucket compatibility.
- Built-in Perchance image, text, and public-gallery APIs under `/perchance/*`,
  including persistent gallery image/URL ID and generator channel settings.

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

Both native generation features require the locally patched sdkit3 build.
WD14 runs in Easy Diffusion's Python process and is backend-independent.

The active conversion matrix is v1↔SDXL, v1↔SD3, SDXL↔SD3, and Flux→v1,
Flux→SDXL, or Flux→SD3. Same-family sources need no converter. Destination
Flux conversions and Stable Cascade are not exposed. Cascade is deferred until
Easy Diffusion and the selected sdkit3 model-loading path can be verified to
support it safely.

## Upstream attribution

- `kohya-ss/ControlNet-LLLite-ComfyUI` — GPL-3.0.
- `city96/SD-Latent-Interposer` — Apache-2.0.
- `pythongosssss/ComfyUI-WD14-Tagger` — MIT.
- WD14 models by SmilingWolf; model-specific terms remain applicable.

The corresponding upstream license texts are retained in `licenses/` in the
deployed project.
