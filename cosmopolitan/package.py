#!/usr/bin/env python3
"""Embed the demo model, UI resources, provenance and notices into the APE."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = {
    "repository": "ggml-org/tiny-llamas",
    "revision": "def3e2dd70df35ecbf6403ea347de4c5977220c1",
    "filename": "stories260K.gguf",
    "sha256": "047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b",
    "upstream_model": "https://huggingface.co/karpathy/tinyllamas",
    "purpose": "Small trained inference fixture; not an image model or a production assistant",
}
MODEL["url"] = (f"https://huggingface.co/{MODEL['repository']}/resolve/"
                f"{MODEL['revision']}/{MODEL['filename']}")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fetch_model(out):
    directory = out / "models"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL["filename"]
    if path.is_file() and digest(path) == MODEL["sha256"]:
        return path
    temporary = path.with_suffix(".part")
    print(f"Fetching pinned inference fixture: {MODEL['repository']}", flush=True)
    request = urllib.request.Request(MODEL["url"], headers={"User-Agent": "EasyDiffusion-Cosmopolitan-Test"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as file:
        shutil.copyfileobj(response, file)
    if digest(temporary) != MODEL["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Inference fixture SHA-256 mismatch")
    temporary.replace(path)
    return path


def source_info():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {"repository": "DemonBigj781/EasyDifussion-Cpp", "commit": commit, "dirty": dirty}


def resources(software_notices=None):
    mappings = [
        (ROOT / "source/UI.cpp/Pages/assets", "cpp-ui/assets"),
        (ROOT / "source/UI.cpp/Pages/src/Plugin/plugin_scripts", "cpp-ui/scripts"),
        (ROOT / "ui/media", "media"),
        (ROOT / "ui/plugins/ui", "plugins/ui"),
        (ROOT / "plugins/ui", "plugins/optional-ui"),
        (ROOT / "easy-diffusion-custom-licenses", "licenses/easy-diffusion-custom"),
        (HERE / "licenses", "licenses/cosmopolitan-integration"),
        (ROOT / "source/llama.cpp/licenses", "licenses/llama-vendor"),
    ]
    if software_notices is not None:
        if not software_notices.is_dir() or not any(software_notices.rglob("*")):
            raise RuntimeError("Missing software WebGPU dependency notices")
        mappings.append((software_notices, "licenses/software-webgpu"))
    result = {}
    for directory, prefix in mappings:
        for file in sorted(directory.rglob("*")):
            if file.is_symlink():
                raise RuntimeError(f"Unexpected resource symlink: {file.relative_to(ROOT)}")
            if file.is_file():
                result[f"{prefix}/{file.relative_to(directory).as_posix()}"] = file
    for source, destination in [
        (ROOT / "LICENSE", "licenses/EasyDiffusion.LICENSE"),
        (ROOT / "THIRD_PARTY_NOTICES.md", "licenses/THIRD_PARTY_NOTICES.md"),
        (ROOT / "source/llama.cpp/LICENSE", "licenses/llama.cpp.LICENSE"),
        (ROOT / "source/sdkit3-port-source/LICENSE", "licenses/sdkit.LICENSE"),
        (ROOT / "source/sdkit3-port-source/stable-diffusion.cpp/LICENSE", "licenses/stable-diffusion.cpp.LICENSE"),
        (ROOT / "source/sdkit3-port-source/stable-diffusion.cpp/ggml/LICENSE", "licenses/diffusion-ggml.LICENSE"),
        (HERE / "MODEL.md", "licenses/MODEL.md"),
        (ROOT / "source/sdkit3-port-source/third_party/SAFETENSORS_CPP_LICENSE.txt", "licenses/safetensors-cpp.LICENSE"),
        (ROOT / "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/LICENSE.darts_clone.txt", "licenses/darts-clone.LICENSE"),
    ]:
        if not source.is_file():
            raise RuntimeError(f"Missing package notice: {source.relative_to(ROOT)}")
        result[destination] = source
    # Preserve notices embedded in the exact vendored headers as distributed.
    # These are documentation resources, not runtime dynamic dependencies.
    header_notices = [
        "source/sdkit3-port-source/third_party/asio.hpp",
        "source/sdkit3-port-source/third_party/base64.hpp",
        "source/sdkit3-port-source/third_party/uuid.h",
        "source/sdkit3-port-source/third_party/crow/TinySHA1.hpp",
        "source/sdkit3-port-source/third_party/crow/http_parser_merged.h",
        "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/stb_image.h",
        "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/stb_image_write.h",
        "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/stb_image_resize.h",
        "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/json.hpp",
        "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/miniz.h",
        "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/zip.h",
        "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/zip.c",
    ]
    for path in header_notices:
        source = ROOT / path
        if not source.is_file():
            raise RuntimeError(f"Missing vendored notice source: {path}")
        result["licenses/source-notices/" + path] = source
    return result


def add(zip_file, name, data, compression=zipfile.ZIP_DEFLATED):
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    zip_file.writestr(info, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "out/easy-diffusion.exe")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--fetch-model-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    model = fetch_model(out)
    if args.fetch_model_only:
        print(model)
        return
    if not args.input or not args.input.is_file():
        parser.error("--input must be the linked Cosmopolitan executable")
    with args.input.open("rb") as file:
        if file.read(2) != b"MZ":
            raise RuntimeError("Expected a Windows-loadable APE input")
    manifest = json.loads(args.metadata.read_text()) if args.metadata else {}
    software_notices = (Path(manifest["software_webgpu"]["licenses"])
                        if "software_webgpu" in manifest else None)
    files = resources(software_notices)
    files["models/" + MODEL["filename"]] = model
    manifest.update({
        "source": source_info(),
        "target": "x86_64 Cosmopolitan APE",
        "model_fixture": MODEL,
        "embedded_resources": {name: {"sha256": digest(file), "bytes": file.stat().st_size}
                               for name, file in files.items()},
    })
    # Preserve the toolchain ZIP resources and the portable shell/PE headers.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".package-", suffix=".exe", dir=out)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(args.input, temporary)
        with zipfile.ZipFile(temporary, "a") as archive:
            duplicates = set(archive.namelist()) & (set(files) | {"BUILD.json"})
            if duplicates:
                raise RuntimeError(f"Input is already packaged: {sorted(duplicates)[:5]}")
            for name, file in files.items():
                compression = zipfile.ZIP_STORED if name.startswith("models/") else zipfile.ZIP_DEFLATED
                add(archive, name, file.read_bytes(), compression)
            add(archive, "BUILD.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
        with zipfile.ZipFile(temporary) as archive:
            if hashlib.sha256(archive.read("models/" + MODEL["filename"])).hexdigest() != MODEL["sha256"]:
                raise RuntimeError("Embedded inference fixture verification failed")
        temporary.chmod(0o755)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest["executable"] = {"filename": args.output.name, "bytes": args.output.stat().st_size,
                              "sha256": digest(args.output)}
    (out / "BUILD.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["executable"], indent=2))


if __name__ == "__main__":
    main()
