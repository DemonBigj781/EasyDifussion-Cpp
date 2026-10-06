#!/usr/bin/env python3
"""Prepare one GGML implementation for the vendored llama and SD sources."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STAMP = "COSMOPOLITAN_SHARED_GGML.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def tracked_entries(source):
    prefix = source.relative_to(ROOT).as_posix() + "/"
    result = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z", "--", prefix], cwd=ROOT)
    entries = []
    for item in result.split(b"\0"):
        if not item:
            continue
        meta, full_path = item.split(b"\t", 1)
        mode, _blob, stage = meta.decode().split()
        if stage != "0" or mode not in ("100644", "100755", "120000"):
            raise RuntimeError("Unsupported or unmerged source entry")
        path = full_path.decode()[len(prefix):]
        entry = {"path": path, "mode": mode}
        entry["sha256"] = content_hash(source, entry)
        entries.append(entry)
    if not entries:
        raise RuntimeError("The authoritative GGML source is not in this Git checkout")
    return entries


def content_hash(directory, entry):
    path = directory / entry["path"]
    if entry["mode"] == "120000":
        if not path.is_symlink():
            raise RuntimeError("Expected source symlink: " + str(path))
        return digest(os.fsencode(os.readlink(path)))
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Expected source file: " + str(path))
    return digest(path.read_bytes())


def prepared_hash(directory, entries):
    return stable_hash([
        {**entry, "sha256": content_hash(directory, entry)} for entry in entries
    ])


def prepare(output):
    pin = json.loads((HERE / "PIN.json").read_text())
    source = ROOT / pin["authoritative_source"]
    entries = tracked_entries(source)
    if (len(entries) != pin["authoritative_file_count"] or
            stable_hash(entries) != pin["authoritative_tree_sha256"]):
        raise RuntimeError("The customized SD GGML tree differs from the reviewed PIN.json")
    for relative, expected in pin["llama_donor_files"].items():
        if digest((ROOT / pin["llama_source"] / relative).read_bytes()) != expected:
            raise RuntimeError("The reviewed llama donor changed: " + relative)
    patches = []
    for name, expected in sorted(pin["patches"].items()):
        patch = HERE / "patches" / name
        if digest(patch.read_bytes()) != expected:
            raise RuntimeError("Shared GGML patch hash mismatch: " + name)
        patches.append(patch)
    adapter_root = HERE.parent / "backend"
    adapter = {
        "source": "cosmopolitan/backend",
        "compile_definition": "GGML_WEBGPU_COSMO",
        "files": {
            path: digest((adapter_root / path).read_bytes())
            for path in ("CMakeLists.txt", "wgpu_adapter.cpp", "include/cosmo-webgpu.h",
                         "include/webgpu/webgpu_cpp.h")
        },
    }
    fingerprint = stable_hash({
        "pin": pin, "prepare_sha256": digest(Path(__file__).read_bytes()),
        "webgpu_adapter": adapter,
    })
    protected = (ROOT / "source", HERE)
    if (output.is_symlink() or output == ROOT or
            any(output == path or output in path.parents or path in output.parents for path in protected)):
        raise RuntimeError("Refusing an unsafe prepared-source destination: " + str(output))
    stamp = output / STAMP
    if stamp.is_file():
        previous = json.loads(stamp.read_text())
        if previous.get("fingerprint") == fingerprint:
            try:
                unchanged = prepared_hash(output, entries) == previous.get("prepared_tree_sha256")
            except (OSError, RuntimeError):
                unchanged = False
            if unchanged:
                print(output)
                return
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Refusing to replace a directory not created by this recipe: " + str(output))

    staged = Path(tempfile.mkdtemp(prefix="." + output.name + "-stage-", dir=output.parent))
    backup = None
    try:
        for entry in entries:
            original = source / entry["path"]
            target = staged / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry["mode"] == "120000":
                target.symlink_to(os.readlink(original))
            else:
                shutil.copy2(original, target)
        for patch in patches:
            subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
                cwd=staged, check=True, stdout=sys.stderr)
        # A changed patch must not make unrelated objects rebuild merely because
        # applying the other patches refreshed their source timestamps.
        if stamp.is_file():
            for entry in entries:
                try:
                    if content_hash(output, entry) == content_hash(staged, entry):
                        shutil.copystat(output / entry["path"], staged / entry["path"], follow_symlinks=False)
                except (OSError, RuntimeError):
                    pass
        metadata = {
            "schema": 1,
            "fingerprint": fingerprint,
            "prepared_tree_sha256": prepared_hash(staged, entries),
            "authoritative_tree_sha256": pin["authoritative_tree_sha256"],
            "authoritative_source": pin["authoritative_source"],
            "authoritative_ggml_base_revision": pin["authoritative_ggml_base_revision"],
            "llama_revision": pin["llama_revision"],
            "ggml_max_name": pin["ggml_max_name"],
            "patches": pin["patches"],
            "webgpu_adapter": adapter,
            "required_compile_definitions": {
                "all_consumers": ["GGML_MAX_NAME=160"],
                "ggml": ["GGML_COSMO_STATIC_ONLY"],
            },
            "extension_backends": {"rope_offset": ["CPU"], "ssm_history": ["CPU"]},
        }
        (staged / STAMP).write_text(json.dumps(metadata, indent=2) + "\n")
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix="." + output.name + "-old-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        staged.rename(output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            backup.rename(output)
        raise
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    if backup is not None:
        shutil.rmtree(backup)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "out/source/shared-ggml")
    args = parser.parse_args()
    # Keep the lock beside the tree so publication cannot replace its inode.
    output = args.output.absolute()
    if output.is_symlink():
        raise RuntimeError("The prepared-source destination must not be a symlink")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".prepare.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("shared GGML preparation failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
