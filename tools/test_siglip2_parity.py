import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()
    assert args.runner.is_file(), "Native SigLIP2 runner has not been implemented"
    args.artifacts.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261002)
    last_pixels = None
    report = {"preprocessing": []}
    for width, height in [(512, 512), (333, 217), (17, 41), (1025, 769), (1, 1024), (1024, 1)]:
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        source = args.artifacts / f"rgb-{width}x{height}.bin"
        output = args.artifacts / f"pixels-{width}x{height}.f32"
        rgb.tofile(source)
        subprocess.run([str(args.runner), "preprocess", str(width), str(height), str(source), str(output)],
                       check=True, timeout=60)
        ratio = 512 / max(width, height)
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        resized = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
        square = Image.new("RGB", (512, 512), (0, 0, 0))
        square.paste(resized, ((512 - size[0]) // 2, (512 - size[1]) // 2))
        expected = (np.asarray(square).astype(np.float32) / 255 - 0.5) / 0.5
        expected = expected.transpose(2, 0, 1).copy()
        actual = np.fromfile(output, dtype="<f4").reshape(3, 512, 512)
        np.testing.assert_array_equal(actual, expected)
        report["preprocessing"].append({"size": [width, height], "max_abs_error": 0})
        if (width, height) == (333, 217):
            last_pixels = output
    if not args.preprocess_only:
        import torch
        from transformers import SiglipVisionModel

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = SiglipVisionModel.from_pretrained(args.model_dir, local_files_only=True,
                                                  attn_implementation="eager").eval().float()
        pixels = torch.from_numpy(np.fromfile(last_pixels, dtype="<f4").reshape(1, 3, 512, 512))
        with torch.inference_mode():
            expected = model(pixels).last_hidden_state.numpy()
        expected.tofile(args.artifacts / "encoder-reference.f32")
        output = args.artifacts / "encoder-native.f32"
        subprocess.run([str(args.runner), "encode", str(args.model_dir / "model.safetensors"),
                        str(last_pixels), str(output)], check=True, timeout=900)
        actual = np.fromfile(output, dtype="<f4").reshape(1, 1024, 768)
        difference = np.abs(actual - expected)
        report["encoder"] = {"max_abs_error": float(difference.max()),
                             "mean_abs_error": float(difference.mean()),
                             "rms_error": float(np.sqrt(np.mean(difference ** 2)))}
        print(json.dumps(report, indent=2), flush=True)
        assert np.isfinite(actual).all()
        assert difference.max() < 0.003, report["encoder"]
        assert report["encoder"]["rms_error"] < 0.0003, report["encoder"]
    (args.artifacts / "parity-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SigLIP2 parity checks passed", flush=True)


if __name__ == "__main__":
    main()
