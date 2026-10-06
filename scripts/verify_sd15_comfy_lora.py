"""Check an exported adapter with the installed ComfyUI loader, without inference."""
import argparse
import json
import logging
import struct
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    sys.argv = [sys.argv[0], "--cpu"]
    import torch
    from safetensors.torch import load_file
    import comfy.lora

    torch.set_num_threads(1)
    with args.model.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    # Only keys/shapes are needed for mapping; do not load a multi-GB base model.
    unet = {key.removeprefix("model."): None for key in header
            if key.startswith("model.diffusion_model.")}
    clip = {key.replace("cond_stage_model.", "clip_l.", 1): None for key in header
            if key.startswith("cond_stage_model.")}
    model = SimpleNamespace(state_dict=lambda: unet,
                            model_config=SimpleNamespace(unet_config={}))
    mapping = comfy.lora.model_lora_keys_unet(model, {})
    mapping = comfy.lora.model_lora_keys_clip(SimpleNamespace(state_dict=lambda: clip), mapping)
    data = load_file(str(args.lora), device="cpu")
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            if "lora key not loaded:" in record.getMessage():
                messages.append(record.getMessage())

    handler = Capture()
    logging.getLogger().addHandler(handler)
    try:
        patches = comfy.lora.load_lora(data, mapping)
    finally:
        logging.getLogger().removeHandler(handler)
    counts = {"unet": sum(k.startswith("diffusion_model.") for k in patches),
              "clip": sum(k.startswith("clip_l.") for k in patches)}
    print(json.dumps({"patches": counts, "unused_keys": len(messages)}))
    assert not messages, f"{len(messages)} unused keys; first: {messages[:1]}"
    assert counts == {"unet": 128, "clip": 48}, counts
    for key, adapter in patches.items():
        up, down, alpha, *_ = adapter.weights
        base_key = ("model." + key if key.startswith("diffusion_model.") else
                    key.replace("clip_l.", "cond_stage_model.", 1))
        shape = header[base_key]["shape"]
        expected = (up @ down) * (alpha / down.shape[0])
        assert list(expected.shape) == shape, (key, expected.shape, shape)
        actual = adapter.calculate_weight(torch.zeros(shape), key, 1.0, 1.0, None,
                                          lambda value: value, torch.float32)
        torch.testing.assert_close(actual, expected)
    print("PASS: all 176 real checkpoint targets mapped and scaled correctly")


if __name__ == "__main__":
    main()
