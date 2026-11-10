# LTX-2.0 and LTX-2.3 in the native video UI

Select the checkpoint under **Options → Model**. Both versions use Gemma 3
12B IT, a video VAE, an audio VAE, and embedding connectors. Their version-specific
companions are not interchangeable. Place checkpoints in the configured
checkpoint/video directory, VAEs in the configured VAE directory, and Gemma
plus connectors in the configured text-encoder directory.

Keep upstream names so automatic selection can identify the resources:

- LTX-2.0 distilled: `LTX2_video_vae_bf16`, `LTX2_audio_vae_bf16`, and
  `ltx-2-19b-embeddings_connector_distill_bf16`.
- LTX-2.3 distilled: `ltx-2.3-22b-distilled_video_vae`,
  `ltx-2.3-22b-distilled_audio_vae`, and
  `ltx-2.3-22b-distilled_embeddings_connectors`.
- Shared text encoder: `gemma-3-12b-it-Q4_K_S` (GGUF).

The VAE and text encoder are under **Options**; the audio VAE and connectors
are under **Video Options**. Overrides are retained per checkpoint. Distilled
checkpoints use Euler, the `ltx2` scheduler, eight steps, and guidance 1.
Missing companions fail before model loading. The UI does not download them
automatically or substitute LTX-2.5 resources.

Quantized weights, CPU offload, mmap, and VAE tiling reduce memory demand but
do not guarantee that a resolution/frame count fits a particular GPU. Start
with a short, low-resolution clip; LTX dimensions are multiples of 32 and
frame counts follow 8n+1. Actual inference must be validated after all weights
finish downloading. The current UI returns numbered video frames; it does not
export generated audio.

See the vendored engine's [LTX guide](../source/sdkit3-port-source/stable-diffusion.cpp/docs/ltx2.md)
for its companion sources and additional engine capabilities.
