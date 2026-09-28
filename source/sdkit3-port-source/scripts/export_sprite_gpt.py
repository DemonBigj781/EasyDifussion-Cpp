#!/usr/bin/env python3
"""Export Sprite-GPT and its CLIP text encoder for the native C++ sidecar."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import CLIPTextModel


class UNetTraceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        half = model.base // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0, dtype=torch.float32))
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        self.register_buffer("timestep_frequencies", frequencies)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        arguments = timestep.float()[:, None] * 1000.0 * self.timestep_frequencies[None]
        embedding = self.model.time_mlp(
            torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        )

        def apply_layer(layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
            if layer.__class__.__name__ == "ResBlock":
                return layer(hidden, embedding)
            if layer.__class__.__name__ == "CrossAttnBlock":
                return layer(hidden, context, context_mask)
            return layer(hidden)

        hidden = self.model.conv_in(x)
        skips = [hidden]
        for block in self.model.down:
            for layer in block:
                hidden = apply_layer(layer, hidden)
            skips.append(hidden)
        for layer in self.model.mid:
            hidden = apply_layer(layer, hidden)
        for block in self.model.up:
            hidden = torch.cat([hidden, skips.pop()], dim=1)
            for layer in block:
                hidden = apply_layer(layer, hidden)
        return self.model.conv_out(torch.nn.functional.silu(self.model.norm_out(hidden)))


class CLIPTraceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto",
                        help="device used consistently for the model and trace inputs")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def resolve_clip_dir(reference: str, offline: bool) -> Path:
    candidate = Path(reference).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return Path(
        snapshot_download(
            reference,
            allow_patterns=[
                "config.json",
                "merges.txt",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
            local_files_only=offline,
        )
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (source_dir / "spritegpt" / "unet.py").is_file():
        raise SystemExit(f"Sprite-GPT source tree not found: {source_dir}")
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    sys.path.insert(0, str(source_dir))
    from spritegpt.unet import UNet

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved.get("config")
    state = saved.get("ema_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise SystemExit("Expected an exported Sprite-GPT checkpoint with config and ema_state_dict")
    if config.get("ctx_dim") != 512 or config.get("num_classes") is not None:
        raise SystemExit("The native exporter currently requires the text-conditioned 512-wide checkpoint")

    model = UNet(
        img_size=int(config["img_size"]),
        base=int(config["base"]),
        ch_mult=tuple(int(value) for value in str(config["ch_mult"]).split(",")),
        num_res=int(config["num_res"]),
        attn_res=tuple(int(value) for value in str(config["attn_res"]).split(",")),
        dropout=0.0,
        num_classes=None,
        ctx_dim=512,
    ).eval()
    model.load_state_dict(state, strict=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    if args.device == "xpu" and not xpu_available:
        raise SystemExit("XPU was requested but Intel XPU PyTorch is not available")
    if args.device == "xpu" or (args.device == "auto" and xpu_available):
        device = torch.device("xpu")
    elif args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    image_size = int(config["img_size"])
    context_length = 32
    torch.manual_seed(0)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    timestep = torch.tensor([0.5], dtype=torch.float32, device=device)
    context = torch.randn(1, context_length, 512, device=device)
    context_mask = torch.ones(1, context_length, dtype=torch.bool, device=device)
    unet_wrapper = UNetTraceWrapper(model).eval().to(device)
    with torch.inference_mode():
        expected_unet = model(x, timestep, None, context, context_mask)
        traced_unet = torch.jit.trace(
            unet_wrapper,
            (x, timestep, context, context_mask),
            strict=False,
            check_trace=False,
        )
        actual_unet = traced_unet(x, timestep, context, context_mask)

    clip_dir = resolve_clip_dir(args.clip, args.offline)
    clip_wrapper = CLIPTraceWrapper(
        CLIPTextModel.from_pretrained(clip_dir, local_files_only=args.offline).eval()
    ).eval().to(device)
    input_ids = torch.full((2, context_length), 49407, dtype=torch.long, device=device)
    input_ids[:, 0] = 49406
    attention_mask = torch.zeros((2, context_length), dtype=torch.long, device=device)
    attention_mask[:, :2] = 1
    with torch.inference_mode():
        expected_clip = clip_wrapper(input_ids, attention_mask)
        traced_clip = torch.jit.trace(
            clip_wrapper,
            (input_ids, attention_mask),
            strict=False,
            check_trace=False,
        )
        actual_clip = traced_clip(input_ids, attention_mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    traced_unet.save(str(output_dir / "sprite-gpt-unet.ts"))
    traced_clip.save(str(output_dir / "clip-text.ts"))
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    for name in (
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        source = clip_dir / name
        if source.is_file():
            shutil.copy2(source, tokenizer_dir / name)

    metadata = {
        "format": "sdkit-sprite-gpt-torchscript-v1",
        "image_size": image_size,
        "channels": 3,
        "context_length": context_length,
        "context_width": 512,
        "train_steps": int(config.get("train_steps", 0)),
        "torch_version": torch.__version__,
        "export_device": str(device),
        "unet_trace_max_abs_error": float((expected_unet - actual_unet).abs().max()),
        "clip_trace_max_abs_error": float((expected_clip - actual_clip).abs().max()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
