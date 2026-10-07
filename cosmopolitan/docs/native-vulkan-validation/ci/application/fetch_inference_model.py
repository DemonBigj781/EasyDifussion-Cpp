#!/usr/bin/env python3
"""Fetch and verify the external trained diffusion checkpoint used by host tests."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE / "PIN.json").read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(path):
    if path.stat().st_size != PIN["bytes"] or digest(path) != PIN["sha256"]:
        raise RuntimeError("Pinned diffusion model size or SHA-256 mismatch: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=HERE.parent / "out/inference/models")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    directory = args.model_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PIN["filename"]
    if path.exists():
        verify(path)
    else:
        if args.check_only:
            raise FileNotFoundError(path)
        curl = shutil.which("curl")
        if not curl:
            raise RuntimeError("The host model fetch requires curl")
        part = path.with_suffix(path.suffix + ".part")
        print("Fetching external trained SD 1.5 checkpoint (1.75 GB)", flush=True)
        subprocess.run([curl, "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                        "--max-time", "1800", "--continue-at", "-", "--output", str(part),
                        PIN["url"]], check=True)
        verify(part)
        part.replace(path)
    print(json.dumps({"path": str(path), "bytes": PIN["bytes"], "sha256": PIN["sha256"]}))


if __name__ == "__main__":
    main()
