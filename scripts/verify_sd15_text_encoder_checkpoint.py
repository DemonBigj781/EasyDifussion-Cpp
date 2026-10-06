"""Verify a joint/frozen two-step native training smoke-test directory."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np


def load(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        data = stream.read()
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        assert entry["dtype"] == "F32", name
        start, end = entry["data_offsets"]
        values = np.frombuffer(data[start:end], dtype="<f4").reshape(entry["shape"])
        assert np.isfinite(values).all(), name
        tensors[name] = values
    return tensors


def verify(directory):
    initial = load(directory / "joint-0.safetensors")
    trained = load(directory / "joint.safetensors")
    frozen = load(directory / "frozen.safetensors")
    assert initial.keys() == trained.keys()
    text = [name for name in trained if name.startswith("lora.cond_stage_model.")]
    unet = [name for name in trained if name.startswith("lora.model.diffusion_model.")]
    assert len(text) == 96 and len(unet) == 256
    assert set(frozen) == set(unet), "frozen encoder must not export text adapters"
    changes = {}
    for group, names in (("text_encoder", text), ("unet", unet)):
        changed = [name for name in names if np.any(trained[name] != initial[name])]
        assert changed and any(name.endswith(".lora_up") for name in changed), group
        assert any(name.endswith(".lora_down") for name in changed), group
        for name in names:
            if name.endswith(".lora_up"):
                assert not np.any(initial[name]), name
        changes[group] = len(changed)
    first_losses = []
    for label in ("joint", "frozen"):
        events = [json.loads(line) for line in (directory / f"{label}.log").read_text().splitlines() if line.startswith('{')]
        assert events[-1]["event"] == "completed"
        first_losses.append(next(event["loss"] for event in events if event["event"] == "progress"))
    assert abs(first_losses[0] - first_losses[1]) < 1e-4, first_losses
    print(json.dumps({"changed_tensors": changes, "first_losses": first_losses, "finite": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
