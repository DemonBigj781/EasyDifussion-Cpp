#!/usr/bin/env python3
"""Build one x86-64 Cosmopolitan application from the vendored source trees."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SDK_VERSION = "4.0.2"
SDK_URL = "https://cosmo.zip/pub/cosmocc/cosmocc-4.0.2.zip"
SDK_SHA256 = "85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44"
LLAMA_REVISION = "c589f0ed10c643678c4707dd160c21ac7633ebc0"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args, **kwargs):
    print("+ " + shlex.join(str(x) for x in args), flush=True)
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def write_if_changed(path, content, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text() != content:
        path.write_text(content)
    if executable:
        path.chmod(0o755)


def sdk_setup(out):
    sdk = out / "sdk"
    stamp = sdk / "ARCHIVE.sha256"
    archive = out / "downloads" / f"cosmocc-{SDK_VERSION}.zip"
    if (stamp.is_file() and stamp.read_text().strip() == SDK_SHA256
            and (sdk / "bin/x86_64-unknown-cosmo-cc").is_file()):
        repair_sdk_links(sdk, archive)
        return sdk
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or digest(archive) != SDK_SHA256:
        temporary = archive.with_suffix(".part")
        print("Fetching pinned Cosmocc " + SDK_VERSION, flush=True)
        with urllib.request.urlopen(SDK_URL, timeout=60) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
        if digest(temporary) != SDK_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Cosmocc archive SHA-256 mismatch")
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe SDK archive member")
    temporary = Path(tempfile.mkdtemp(prefix="sdk-", dir=out))
    run(["unzip", "-q", archive, "-d", temporary])
    if sdk.exists():
        shutil.rmtree(sdk)
    temporary.replace(sdk)
    stamp.write_text(SDK_SHA256 + "\n")
    repair_sdk_links(sdk, archive)
    return sdk


def repair_sdk_links(sdk, archive):
    # Some archive extractors materialize Unix links as plain text files.
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            mode = member.external_attr >> 16
            path = sdk / member.filename
            if mode & 0o170000 == 0o120000:
                target = zipped.read(member).decode()
                if not (path.parent / target).resolve().is_relative_to(sdk.resolve()):
                    raise RuntimeError("SDK link escapes its directory")
                if not path.is_symlink() or os.readlink(path) != target:
                    path.unlink(missing_ok=True)
                    path.symlink_to(target)
            elif path.is_file() and not path.is_symlink():
                path.chmod(mode & 0o777)


def host_tools(out):
    tools = out / "tools"
    cmake = tools / "bin/cmake"
    ninja = tools / "bin/ninja"
    if not cmake.is_file() or not ninja.is_file():
        if not (tools / "bin/python").is_file():
            run([sys.executable, "-m", "venv", tools])
        run([tools / "bin/python", "-m", "pip", "install", "--disable-pip-version-check",
             "cmake==3.31.10", "ninja==1.13.0"])
    return cmake, ninja


def make_wrappers(out, sdk):
    directory = out / "toolchain"
    loader = sdk / "bin/ape-x86_64.elf"
    for name, tool in {
        "cc": "x86_64-unknown-cosmo-cc", "cxx": "x86_64-unknown-cosmo-c++",
        "ar": "x86_64-unknown-cosmo-ar", "ranlib": "x86_64-linux-cosmo-ranlib",
        "strip": "x86_64-unknown-cosmo-strip",
    }.items():
        binary = sdk / "bin" / tool
        content = ("#!" + sys.executable + "\nimport os,sys\n"
                   + "binary=" + repr(str(binary)) + "\n"
                   + "loader=" + repr(str(loader)) + "\n"
                   + "with open(binary,'rb') as f: magic=f.read(4)\n"
                   + "argv=([loader,binary] if magic[:2]==b'MZ' else [binary])+sys.argv[1:]\n"
                   + "os.execv(argv[0],argv)\n")
        write_if_changed(directory / name, content, executable=True)
    return directory


def prepare_source(out):
    paths = ["source/llama.cpp", "source/sdkit3-port-source", "source/UI.cpp"]
    names = subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=ROOT).split(b"\0")
    names = [x.decode() for x in names if x]
    identity = []
    for name in names:
        path = ROOT / name
        if path.is_symlink():
            identity.append((name, "symlink", os.readlink(path)))
        else:
            identity.append((name, "file", digest(path)))
    patches = sorted((HERE / "patches").glob("*.patch")) + sorted((HERE / "app-patches").glob("*.patch"))
    fingerprint = hashlib.sha256(json.dumps({
        "source": identity, "patches": [(p.name, digest(p)) for p in patches],
    }, sort_keys=True).encode()).hexdigest()
    source = out / "source"
    stamp = source / "application.fingerprint"
    if not stamp.is_file() or stamp.read_text().strip() != fingerprint:
        source.mkdir(parents=True, exist_ok=True)
        for path in paths:
            dest = source / Path(path).name
            if dest.exists():
                shutil.rmtree(dest)
        for name in names:
            original = ROOT / name
            target = source / Path(name).relative_to("source")
            target.parent.mkdir(parents=True, exist_ok=True)
            if original.is_symlink():
                target.symlink_to(os.readlink(original))
            else:
                shutil.copy2(original, target)
        for patch in patches:
            working = source / "sdkit3-port-source" if patch.parent.name == "app-patches" else source
            run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", patch], cwd=working)
        stamp.write_text(fingerprint + "\n")
    shared = source / "shared-ggml"
    prepare = HERE / "shared-ggml/prepare.py"
    if not prepare.is_file():
        raise RuntimeError("Shared GGML preparation script is not ready: " + str(prepare))
    run([sys.executable, prepare, "--output", shared])
    return source, shared, fingerprint


def set_stack(path):
    with path.open("r+b") as file:
        if file.read(2) != b"MZ":
            raise RuntimeError("Expected a PE-capable APE")
        file.seek(0x3c)
        pe = struct.unpack("<I", file.read(4))[0]
        file.seek(pe)
        if file.read(4) != b"PE\0\0":
            raise RuntimeError("Invalid PE signature")
        file.seek(pe + 24)
        if struct.unpack("<H", file.read(2))[0] != 0x20b:
            raise RuntimeError("Expected PE32+ optional header")
        file.seek(pe + 24 + 72)
        file.write(struct.pack("<QQ", 8 * 1024 * 1024, 4096))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("COSMO_BUILD_JOBS", "2")))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--libraries-only", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "build.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sdk = sdk_setup(out)
    cmake, ninja = host_tools(out)
    wrappers = make_wrappers(out, sdk)
    source, shared, fingerprint = prepare_source(out)
    shutil.copy2(sdk / "bin/ape-x86_64.elf", out / "ape-x86_64.elf")
    if args.prepare_only:
        print("Prepared source: " + str(source))
        return
    env = dict(os.environ, COSMO_SDK=str(sdk), COSMO_BUILD_TOOLS=str(wrappers))
    env["PATH"] = str(wrappers) + os.pathsep + str(out / "tools/bin") + os.pathsep + env["PATH"]
    build = out / "build"
    run([cmake, "-S", HERE, "-B", build, "-G", "Ninja",
         "-DCMAKE_TOOLCHAIN_FILE=" + str(HERE / "cmake/cosmocc.cmake"),
         "-DCMAKE_MAKE_PROGRAM=" + str(ninja), "-DCMAKE_BUILD_TYPE=Release",
         "-DCOSMO_SOURCE_ROOT=" + str(source), "-DCOSMO_SHARED_GGML=" + str(shared),
         "-DCOSMO_BUILD_APP=" + ("OFF" if args.libraries_only else "ON")], env=env)
    targets = ["llama", "stable-diffusion", "easy-diffusion-ui"] if args.libraries_only else ["easy-diffusion"]
    run([cmake, "--build", build, "--parallel", args.jobs, "--target", *targets], env=env)
    if args.libraries_only:
        print("Shared GGML, llama.cpp, stable-diffusion.cpp, and UI static targets built")
        return
    debug = build / "bin/easy-diffusion.com.dbg"
    linked = out / "easy-diffusion.unpacked.exe"
    run([sdk / "bin/ape-x86_64.elf", sdk / "bin/apelink", "-o", linked,
         "-l", sdk / "bin/ape-x86_64.elf", debug])
    set_stack(linked)
    shutil.copy2(debug, out / "easy-diffusion.com.dbg")
    metadata = {"cosmocc": {"version": SDK_VERSION, "url": SDK_URL, "sha256": SDK_SHA256},
                "llama_revision": LLAMA_REVISION, "source_fingerprint": fingerprint,
                "cpu_baseline": "x86-64; no AVX requirement", "shared_ggml": True,
                "openmp": False, "dynamic_backends": False,
                "vocabulary_translation_units": 12}
    recipe_files = [HERE / "build.py", HERE / "build.sh", HERE / "CMakeLists.txt"]
    for directory in ("cmake", "patches", "app-patches"):
        recipe_files.extend(p for p in (HERE / directory).rglob("*") if p.is_file())
    metadata["build_recipe_sha256"] = {
        str(p.relative_to(HERE)): digest(p) for p in sorted(recipe_files)}
    shared_metadata = shared / "COSMOPOLITAN_SHARED_GGML.json"
    if shared_metadata.is_file():
        metadata["shared_ggml_metadata"] = json.loads(shared_metadata.read_text())
    manifest = out / "LINK.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if not args.skip_package:
        run([sys.executable, HERE / "package.py", "--input", linked,
             "--output", out / "easy-diffusion.exe", "--metadata", manifest])
    print("Linked application: " + str(linked))


if __name__ == "__main__":
    main()
