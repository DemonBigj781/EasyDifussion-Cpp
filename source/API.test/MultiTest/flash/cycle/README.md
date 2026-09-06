# FlashAttention API roundtrip cycle test

`flash-api-roundtrip-cuda-test.cu` is active and is built by the
`flash-cuda-roundtrip` target in `source/API.test/Makefile`.

Each of 32 cycles with the default fixture verifies both directions of the
normalized API route:

1. model fixture bytes enter through the Common CUDA load route;
2. the device-owned bytes return to the test unchanged;
3. a normalized FlashAttention request enters Common, the registered CUDA
   translation and CUDA definition, and the device kernel;
4. the resulting tensor returns through Common and matches the analytical
   result;
5. the model resource exits through Common unload and is cleared.

This is a roundtrip API test, not an image-generation or Easy Diffusion
end-to-end test. The CPU native-boundary smoke test remains under
`Feature/Attention/Flash/Cpu`.

A user-supplied fixture larger than 64 MiB runs one cycle so a real model does
not cause 32 redundant GPU uploads.

Build and run from `source/API.test` with:

```text
make flash-cuda-roundtrip
build/flash-api-roundtrip-cuda-test MODELS/lifecycle-model.fixture
```
