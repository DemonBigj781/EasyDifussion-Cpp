# FlashAttention image-generation end-to-end test

`flash-cuda-gen.cpp` is active and built with `make flash-cuda-gen` from
`source/API.test`. It generates one 512x512 image through stable-diffusion.cpp
and the normalized Flash Common CUDA route. Sampling is fixed to
`ddim_trailing` with the `simple` schedule.

The executable selects CUDA device 0 by default, enables
`SD_CUDA_FLASH_COMMON=1`, disables the xFormers override, and samples the GGML
application adapter's Common launch counter. It fails if generation completes
without a Common Flash launch, so the optimized `fattn` compatibility fallback
cannot silently satisfy the test.

```text
make flash-cuda-gen
build/flash-cuda-gen \
  -m MODELS/Checkpoint/sd-v1-5.safetensors \
  -v MODELS/Vae/vae-ft-mse-840000-ema-pruned.safetensors \
  -p "a lighthouse above a stormy sea" \
  -n "blurry, low quality" \
  -s 1 -c 7 -d 0 \
  -o build/flash-cuda-smoke.ppm
```

The local RTX 3060 validation generated a 512x512 image in 26.7 seconds and
recorded 40 normalized Common Flash launches. The sibling `../cycle/` test
remains the repeated request/result and model-byte lifecycle API test.
