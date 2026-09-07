# xFormers image-generation test

These executables generate one 512x512 image with xFormers-compatible memory-efficient attention. They use only the `ddim_trailing` sampler and its canonical `simple` schedule; sampler and scheduler overrides are intentionally not exposed.

## Build

From `source/API.test`:

```sh
make xformers-cpu-gen
make xformers-cuda-gen
```

The Makefile configures an isolated static stable-diffusion.cpp build for the selected backend. Outputs are `build/xformers-cpu-gen` and `build/xformers-cuda-gen`.

## Options

```text
-p TEXT   positive prompt; required
-n TEXT   negative prompt; empty by default
-s N      DDIM step count; 20 by default
-m PATH   checkpoint/model path; required
-v PATH   optional external VAE path
-d VALUE  device id, UUID, or name
-c VALUE  CFG scale; 7.0 by default
-o PATH   output PPM path
-h        help
```

CUDA device selection accepts an integer such as `0`, a backend id such as `cuda0`, a full `GPU-...` UUID, or a unique case-insensitive substring of the CUDA device name. CPU accepts `cpu`, `cpu0`, or `0`.

Example CUDA run:

```sh
./build/xformers-cuda-gen \
  -m MODELS/Checkpoint/sd-v1-5.safetensors \
  -v MODELS/Vae/vae-ft-mse-840000-ema-pruned.safetensors \
  -p "a lighthouse above a stormy sea" \
  -n "blurry, low quality" \
  -s 20 -c 7 -d 0 \
  -o xformers-cuda-output.ppm
```

The CUDA program enables `SD_CUDA_XFORMERS=1`, disables the Flash Common override, and enables diffusion/general flash attention before loading the model. It samples the native CUDA xFormers launch counter around generation and fails if no xFormers launch occurred, so an unsupported operation cannot silently fall back and still produce a passing test. The CPU program requests stable-diffusion.cpp's reference CPU memory-efficient attention path. Its argument, image-writing, and static stable-diffusion.cpp build harness is also shared by the sibling Flash generator.

The sibling `cycle/` executables separately validate the project Common xFormers contract together with the custom Common model load/unload handlers. Image context loading remains owned by stable-diffusion.cpp because it parses and constructs model tensors rather than merely retaining an opaque byte copy.

When no CUDA architecture is supplied to CMake, the isolated CUDA build targets compute capability 8.0 and emits forward-compatible PTX. The xFormers kernel itself supports Pascal, but this integration executable links the complete ggml-CUDA target, which also compiles the SM80 SageAttention translation. Set `CMAKE_CUDA_ARCHITECTURES` explicitly when a deployment requires architecture-specific code and ensure every CUDA translation in the combined target supports it.
