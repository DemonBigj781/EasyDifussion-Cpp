# sdkit
**sdkit** (**s**table **d**iffusion **kit**) is an easy-to-use library for using Stable Diffusion in your AI Art projects. It is fast, feature-packed, and memory-efficient.

[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB)

WORK IN PROGRESS on v3 - uses [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) under-the-hood.

If you wish to integrate this into your commercial project or use it on your inference servers, please feel free to contact me (cmdr2 on Discord, or dev@cmdr2.org).

## Native WD14 tagger

`sdkit-wd14-tagger` implements WD14 ONNX inference in C++. It keeps the ONNX
session loaded while the Easy Diffusion tagging API streams image requests to
it. The tagger build is enabled automatically when the repository `.venv`
contains the ONNX Runtime shared library. Otherwise configure
`SDKIT_BUILD_WD14_TAGGER=ON` and set `SDKIT_ONNXRUNTIME_LIBRARY` to an ONNX
Runtime 1.23.2 compatible shared library. ONNX Runtime API headers and license
are in `third_party/onnxruntime`.

The API uses the native sidecar when it is available. Set
`SDKIT_WD14_TAGGER=/path/to/sdkit-wd14-tagger` to select a deployed executable;
without the sidecar, the existing Python inference path remains available.
This native build currently uses ONNX Runtime's CPU provider.

## Native Sprite-GPT sidecar

`sdkit-sprite-gpt` provides native C++ inference for the text-conditioned
[Sprite-GPT](https://huggingface.co/gmmeyer/sprite-gpt) checkpoint. It includes
CLIP byte-pair tokenization, classifier-free guidance, Euler rectified-flow
sampling, deterministic seeds, CPU/CUDA execution, and PNG output. The model
weights remain external to this repository.

The native tokenizer requires ICU development headers and libraries.

Export the upstream checkpoint and CLIP encoder once. The exporter writes two
TorchScript model containers plus tokenizer metadata to the chosen artifact
directory:

```sh
pyenv shell comfyui
python source/sdkit3-port-source/scripts/export_sprite_gpt.py \
    --source-dir "$SPRITE_GPT_SOURCE" \
    --checkpoint "$SPRITE_GPT_SOURCE/sprite-gpt-text64.pt" \
    --clip openai/clip-vit-base-patch32 \
    --output-dir "$SPRITE_GPT_ARTIFACTS"
```

Use `--offline` when `--clip` points to an existing Hugging Face snapshot.
Build against the PyTorch installation in the active pyenv environment:

```sh
SDKIT_TORCH_PYTHON="$(pyenv which python)" \
    ./source/Build.sh --sprite-gpt --target sdkit-sprite-gpt
```

Generate a 64-by-64 sprite:

```sh
BUILD_DIR=source/sdkit3-port-source/build/local-linux-x64-cuda-sm86
"$BUILD_DIR/bin/sdkit-sprite-gpt" \
    --unet "$SPRITE_GPT_ARTIFACTS/sprite-gpt-unet.ts" \
    --clip "$SPRITE_GPT_ARTIFACTS/clip-text.ts" \
    --tokenizer "$SPRITE_GPT_ARTIFACTS/tokenizer" \
    --prompt "a blue ghost sprite" \
    --output blue-ghost.png \
    --steps 50 \
    --guidance 5 \
    --seed 42 \
    --device cuda
```

The executable prints a JSON result on success. Pass `--device cpu` for CPU
inference or `--tokenize-only` to inspect CLIP token IDs without loading model
weights. The build records the selected LibTorch paths in the executable's
build RPATH; set `LD_LIBRARY_PATH` or bundle the matching libraries if the
binary is moved elsewhere.

The upstream checkpoint was trained from multiple sprite and emoji datasets.
Review the upstream model card and its dataset licenses before redistributing
weights or generated assets.


### Loading Sprite-GPT in Easy Diffusion

Deploy `sdkit-sprite-gpt` beside the `sdkit` executable in the selected backend
directory. Add a `*.sprite-gpt.json` file beneath a configured checkpoint folder:

```json
{
  "format": "sdkit-sprite-gpt-torchscript-v1",
  "image_size": 64,
  "unet": "/path/to/native/sprite-gpt-unet.ts",
  "clip": "/path/to/native/clip-text.ts",
  "tokenizer": "/path/to/native/tokenizer"
}
```

Artifact paths may also be relative to the manifest. Refresh the model list
and select the entry labeled **Sprite-GPT 64×64**. The normal Make Image flow
uses the native generator, with fixed 64×64 output and Euler rectified-flow
sampling. Prompt, negative prompt, seed, steps, guidance, batches, output
format, progress and cancellation are supported. The bundled CLIP encoder is
used automatically; no VAE is needed. Initial/reference images, LoRAs,
embeddings, ControlNet, latent interposers and tiling are unsupported and are
rejected if supplied. CPU and CUDA compute devices are supported.

The native CLI accepts `--negative-prompt TEXT` and `--progress`. Progress
emits one JSON line per sampling step before the final result JSON.
