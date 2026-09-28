# LTX-Video 0.9.x

The original video-only LTX-Video transformer is supported separately from the newer
LTX-2 audio/video architecture. This includes the official 2B distilled checkpoints,
such as `ltxv-2b-0.9.6-distilled-04-25.safetensors` and
`ltxv-2b-0.9.8-distilled.safetensors`.

## Models

- Download a checkpoint from [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video).
  The official single-file checkpoints contain both the transformer and video VAE.
- Download a T5-XXL text encoder compatible with the checkpoint. LTX-Video uses a
  maximum prompt length of 256 tokens.

## Distilled inference

The distilled 2B models use Euler sampling without classifier-free guidance. Eight
steps with the `linear_quadratic` scheduler reproduce the model's native schedule.
The scheduler is selected automatically for detected LTX-Video 0.9.x checkpoints,
but it is included explicitly below for clarity.

In Easy Diffusion, select the checkpoint and T5-XXL file under **Video Options** and
leave the VAE set to **Auto-detect / embedded**. Checkpoints whose filenames contain
`distilled` automatically select Euler, `linear_quadratic`, eight steps, and guidance 1.

```bash
./bin/sd-cli -M vid_gen \
  -m ../models/ltxv-2b-0.9.8-distilled.safetensors \
  --t5xxl ../models/t5xxl_fp16.safetensors \
  -p "A cinematic tracking shot of a fox running through fresh snow" \
  --cfg-scale 1.0 \
  --sampling-method euler \
  --scheduler linear_quadratic \
  --steps 8 \
  -W 768 -H 512 \
  --video-frames 97 \
  --fps 24 \
  --diffusion-fa \
  -o ltxv.webm
```

Use `-DSD_CUDA=ON` for a CUDA build or `-DSD_VULKAN=ON` for a Vulkan build. Both
backends execute the same LTX-Video graph and support full-precision safetensors as
well as converted GGUF transformer weights.

LTX-Video's VAE has a spatial scale factor of 32 and a temporal scale factor of 8,
so output dimensions should be divisible by 32 and frame counts should normally be
`8n + 1`.
