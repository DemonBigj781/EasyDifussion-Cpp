# Native model directory arguments

Use `--dir-<model-type> <dir>` when starting the project's native `sdkit`
binary directly. These arguments set the model discovery directories, not
individual files.

| New option | Retired direct CLI spelling |
| --- | --- |
| `--dir-stable-diffusion` | `--ckpt-dir` |
| `--dir-vae` | `--vae-dir` |
| `--dir-hypernetwork` | `--hypernetwork-dir` |
| `--dir-gfpgan` | `--gfpgan-models-path` |
| `--dir-realesrgan` | `--realesrgan-models-path` |
| `--dir-lora` | `--lora-dir` |
| `--dir-codeformer` | `--codeformer-models-path` |
| `--dir-embeddings` | `--embeddings-dir` |
| `--dir-controlnet` | `--controlnet-dir` |
| `--dir-text-encoder` | `--text-encoder-dir` |

For example:

```sh
sdkit --dir-stable-diffusion "/models/checkpoints" --dir-lora "/models/my LoRAs"
```

The old native spellings fail with their specific replacement, rather than
quietly aliasing a new flag. Existing settings keys and folders are unchanged.
When launching through EasyDiffusion, edit model directories through the model
directory settings, not `backend_config.COMMANDLINE_ARGS`. The launcher probes
its installed bundle's help to select a complete directory vocabulary and can
still send the legacy spelling to a legacy release. Both sets are blocked from
custom startup args so they can't override the managed directories.

`--control-net-sd1-path` and `--control-net-sdxl-path` still take individual
weights files. For a directory of discoverable models, use `--dir-controlnet`.

The direct Xavier benchmark and isolated LTX native API test target the
project-built binary and use this new vocabulary.

For a model-free native smoke test:

```sh
python3 scripts/test_native_model_directory_cli.py --binary /path/to/sdkit
```

Pass `--runtime /path/to/lib` if the shared libraries are not next to the binary.
