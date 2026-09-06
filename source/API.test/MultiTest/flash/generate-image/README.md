# FlashAttention image-generation end-to-end test

No image-generation target is implemented here yet. This directory is separate
from the active API roundtrip test under `../cycle/`.

Promotion requires a stable-diffusion.cpp adapter that converts the live GGML
FlashAttention operation into the normalized Common request and returns through
the same route. The current optimized GGML `fattn` compatibility path does not
call Common, so running an image through it would not prove the intended API
integration.
