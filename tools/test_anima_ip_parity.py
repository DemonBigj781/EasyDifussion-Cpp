import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    assert args.runner.is_file(), "Native Anima IP attention runner has not been implemented"
    args.artifacts.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    weights = load_file(args.model)
    query = torch.randn(1, 7, 2048)
    tokens = torch.randn(1, 19, 768)
    embedding = torch.randn(1, 1, 2048)
    for name, value in [("query", query), ("tokens", tokens), ("embedding", embedding)]:
        value.numpy().tofile(args.artifacts / (name + ".f32"))
    report = []
    for block in [0, 13, 27]:
        def linear(name, value):
            prefix = f"blocks.{block}.{name}"
            return F.linear(value, weights[prefix + ".weight"].float(), weights[prefix + ".bias"].float())

        def lora(name, value):
            prefix = f"lora.base_model.model.blocks.{block}.cross_attn.{name}"
            return F.linear(F.linear(value, weights[prefix + ".lora_A.weight"].float()),
                            weights[prefix + ".lora_B.weight"].float())

        for strength in [0.0, 0.35, 1.0]:
            q = query.view(1, 7, 16, 128).transpose(1, 2)
            k = linear("ip_k_proj", tokens * strength).view(1, 19, 16, 128).transpose(1, 2)
            v = linear("ip_v_proj", tokens * strength).view(1, 19, 16, 128).transpose(1, 2)
            expected = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 7, 2048)
            expected *= linear("adaln_ip.1", F.silu(embedding))
            if strength == 0:
                expected.zero_()
            output = args.artifacts / f"block-{block}-{strength}.f32"
            subprocess.run([str(args.runner), str(args.model), str(args.artifacts), str(block), str(strength), str(output)],
                           check=True, timeout=180)
            actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
            difference = np.abs(actual - expected.numpy())
            report.append({"block": block, "strength": strength, "max_abs_error": float(difference.max())})
            np.testing.assert_allclose(actual, expected.numpy(), atol=0.001, rtol=0.0002)
            if strength == 0:
                np.testing.assert_array_equal(actual, expected.numpy())
        for name, value in [("q_proj", query), ("k_proj", query[:, :, :1024].contiguous()),
                            ("v_proj", query[:, :, :1024].contiguous()), ("output_proj", query)]:
            expected = lora(name, value)
            actual = np.fromfile(args.artifacts / f"lora-{block}-{name}.f32", dtype="<f4").reshape(expected.shape)
            np.testing.assert_allclose(actual, expected.numpy(), atol=0.0001, rtol=0.0001)
    print(json.dumps(report, indent=2), flush=True)
    (args.artifacts / "parity-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
