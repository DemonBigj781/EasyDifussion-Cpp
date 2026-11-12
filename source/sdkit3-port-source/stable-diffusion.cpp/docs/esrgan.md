## Using ESRGAN to upscale results

You can use ESRGAN—such as the model [RealESRGAN_x4plus_anime_6B.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)—to upscale the generated images and improve their overall resolution and clarity.

RGBA inputs retain their alpha channel during model upscaling and hires fix. ESRGAN processes RGB only; alpha is resized with bilinear interpolation and recombined with the result.

- Specify the model path using the `--upscale-model PATH` parameter. example:

```bash
sd-cli -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat" --upscale-model ../models/RealESRGAN_x4plus_anime_6B.pth
```
