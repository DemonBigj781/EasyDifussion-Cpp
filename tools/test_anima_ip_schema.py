import argparse
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    with args.model.open("rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        original = json.loads(source.read(length))
    payload_size = args.model.stat().st_size - length - 8
    cases = []
    for key, value in [("ip_norm_keys", "True"), ("ip_inject_before_mlp", "True"),
                       ("lora_rank", "16"), ("lora_alpha", "16")]:
        changed = copy.deepcopy(original)
        changed["__metadata__"][key] = value
        cases.append((key, changed, "Unsupported Anima IP metadata: " + key))
    changed = copy.deepcopy(original)
    del changed["blocks.27.ip_v_proj.weight"]
    cases.append(("missing-block", changed, "Missing Anima IP tensor"))
    changed = copy.deepcopy(original)
    changed["blocks.0.ip_k_proj.weight"]["shape"] = [768, 2048]
    cases.append(("wrong-shape", changed, "Unsupported Anima IP shape"))
    changed = copy.deepcopy(original)
    changed["unrecognized.weight"] = {"dtype": "BF16", "shape": [1], "data_offsets": [payload_size, payload_size + 2]}
    cases.append(("unknown-weight", changed, "Unsupported Anima IP tensor"))
    with tempfile.TemporaryDirectory(prefix="anima-schema-") as directory:
        for name, header, message in cases:
            path = Path(directory) / (name + ".safetensors")
            encoded = json.dumps(header).encode()
            with path.open("wb") as output:
                output.write(struct.pack("<Q", len(encoded)))
                output.write(encoded)
                output.truncate(8 + len(encoded) + payload_size + 2)
            result = subprocess.run([str(args.runner), str(path), directory, "0", "1", str(Path(directory) / "output")],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            assert result.returncode != 0 and message in result.stdout, (name, result.returncode, result.stdout)
            print("PASS:", name)


if __name__ == "__main__":
    main()
