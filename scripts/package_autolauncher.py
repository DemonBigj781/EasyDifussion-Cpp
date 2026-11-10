"""Hash an already staged bundle and publish its autolauncher manifest.

Build-time tool only: no downloads, Python environment creation, or application
execution. Use `uv run --no-project python scripts/package_autolauncher.py ...`
or an existing build Python interpreter.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def staged_path(root, name, directory=False):
    if not isinstance(name, str) or not name or any(c in name for c in "\\:\0"):
        raise ValueError(f"Invalid relative path: {name!r}")
    if name != "." or not directory:
        if any(part in ("", ".", "..") for part in name.split("/")):
            raise ValueError(f"Unsafe relative path: {name!r}")
    path = (root / name).resolve(strict=True)
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes bundle: {name}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Wrong file type: {name}")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_manifest(spec, root):
    root = Path(root).resolve(strict=True)
    if not isinstance(spec, dict) or set(spec) != {"schema", "bundles"} or type(spec["schema"]) is not int or spec["schema"] != 1:
        raise ValueError("Expected schema 1 and bundles")
    result = copy.deepcopy(spec)
    if not isinstance(result["bundles"], list) or not 1 <= len(result["bundles"]) <= 128:
        raise ValueError("Expected 1..128 bundles")
    ids = set()
    for bundle in result["bundles"]:
        if bundle["id"] in ids:
            raise ValueError("Duplicate bundle id")
        ids.add(bundle["id"])
        if "files" in bundle:
            raise ValueError("Packaging specs use assets, not precomputed files hashes")
        assets = bundle.pop("assets", [])
        if not isinstance(assets, list) or not all(isinstance(p, str) for p in assets):
            raise ValueError("assets must be an array of paths")
        paths = set(assets)
        programs = bundle["programs"]
        if not isinstance(programs, list) or not 1 <= len(programs) <= 16:
            raise ValueError("Expected 1..16 programs")
        for i, command in enumerate(programs):
            if command["role"] != ("main" if i == len(programs) - 1 else "service"):
                raise ValueError("Services must precede one main program")
        commands = programs + ([bundle["probe"]] if "probe" in bundle else [])
        if bundle["backend"] != "cpu" and "probe" not in bundle:
            raise ValueError("Accelerated bundles require a compatibility probe")
        for command in commands:
            paths.add(command["executable"])
            staged_path(root, command.get("cwd", "."), directory=True)
        bundle["files"] = {name: file_hash(staged_path(root, name)) for name in sorted(paths)}
    return result


def publish(root, manifest, launcher, replace=False):
    output = root / "launcher.json"
    target = root / launcher.name
    same_launcher = target.exists() and target.resolve() == launcher.resolve()
    if not replace and (output.exists() or (target.exists() and not same_launcher)):
        raise ValueError("Output already exists; use --replace for an intentional repack")
    if any(launcher.name in b["files"] or "launcher.json" in b["files"] for b in manifest["bundles"]):
        raise ValueError("Launcher outputs cannot also be worker assets")
    temporary = []
    try:
        if not same_launcher:
            fd, name = tempfile.mkstemp(prefix=".autolauncher-", dir=root)
            os.close(fd)
            staged_launcher = Path(name)
            temporary.append(staged_launcher)
            shutil.copy2(launcher, staged_launcher)
        fd, name = tempfile.mkstemp(prefix=".manifest-", dir=root)
        temporary.append(Path(name))
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not same_launcher:
            os.replace(staged_launcher, target)
        os.replace(name, output)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return target, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        launcher = args.launcher.resolve(strict=True)
        if not root.is_dir() or not launcher.is_file():
            raise ValueError("Expected staged root directory and compiled launcher file")
        if args.spec.resolve() == root / "launcher.json":
            raise ValueError("Keep the source specification separate from launcher.json")
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        manifest = make_manifest(spec, root)
        target, output = publish(root, manifest, launcher, args.replace)
        print(f"Packaged {target}\nManifest: {output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Packaging failed: {error}\n")


if __name__ == "__main__":
    main()
