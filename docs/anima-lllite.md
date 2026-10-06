# Anima ControlNet-LLLite

The native backend supports the **RGB-conditioning** Anima LLLite v2 architecture.
Both interfaces offer compatible Anima weights when an Anima checkpoint is selected.
UNet LLLite weights are kept separate. Unknown architectures and unsupported masked
inpainting weights are excluded from selection.

## Use

1. Select an Anima checkpoint, its Qwen Image VAE, and Qwen3-0.6B text encoder.
2. Set **ControlNet → Mode → ControlNet-LLLite**.
3. Select `Anima/kohya-ss/base-v1/anima-lllite-any-test-like-v2` to start.
4. Upload a prepared conditioning image. The modern UI does not automatically
   extract depth, pose, or lineart; supply the appropriate condition yourself.
5. Start with strength `1`, start `0%`, and end `100%`, then generate normally.

The model and numeric controls participate in the modern input-saving preferences.

## Obtained weights

Stored under `models/controlnet-lllite/Anima/kohya-ss/`:

| Folder | Weight | Status |
|---|---|---|
| `base-v1` | `anima-lllite-any-test-like-v2` | RGB conditioning supported |
| `base-v1` | `anima-lllite-inpainting-v2` | Downloaded; separate four-channel mask path not implemented |
| `preview3` | `anima-lllite-lineart-1` | RGB conditioning supported |
| `preview3` | `anima-lllite-depth-1` | RGB conditioning supported |
| `preview3` | `anima-lllite-pose-1` | RGB conditioning supported |
| `preview3` | `anima-lllite-scribble-1` | RGB conditioning supported |

The Preview3 weights were trained against an older Anima base; the author's model
card describes reduced quality on Anima-Base v1.0. The weights have the included
CircleStone Labs Non-Commercial License. Preserve their `LICENSE`, `README.md`,
`PREVIEW3.md`, and checksum/provenance manifest.

Source: `kohya-ss/Anima-LLLite`, revision
`36ba7f2f498a1ca63cc77fc7a1298652f72d4524`.
Reference implementation: `kohya-ss/sd-scripts`,
`networks/control_net_lllite_anima.py`.

## Verification and limitations

- All six downloads match the published SHA-256 digests and pass tensor-bound checks.
- Independent PyTorch-reference parity passed on CPU and CUDA for the shared
  conditioning encoder and the first attention-input residual.
- Actual Anima-Base v1.0 generation passed: two different conditions give identical
  images at zero strength and different images at nonzero strength.
- The production Easy Diffusion `/render` request path returned a successful image.
- Both UIs were checked against the actual installed models; each offers five RGB
  weights and excludes the four-channel inpainting model.
- No ASPP or concatenated reference-image support is claimed. Vulkan numerical
  parity and visual quality across every weight/checkpoint combination are unverified.

Regression sources are in `tools/test_anima_lllite_metadata.py`,
`tools/test_anima_lllite_reference.py`, `scripts/test_anima_lllite_ui.cjs`, and
`source/sdkit3-port-source/stable-diffusion.cpp/tests/anima_lllite_parity.cpp`.
