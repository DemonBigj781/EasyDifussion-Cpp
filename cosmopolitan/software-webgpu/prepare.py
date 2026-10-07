#!/usr/bin/env python3
"""Build the pinned Cosmopolitan WGPU, Mesa and LLVM static dependency closure."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
PORT = HERE.parent
PIN = json.loads((HERE / "PIN.json").read_text())
SDK_SHA = "85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text() != content:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)


def source_inventory(root):
    files = []
    for name in PIN["paths"]:
        path = root / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
        else:
            raise RuntimeError("Missing foundation source: " + str(path))
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def application_patches():
    # Additive C ABI fixes belong to this application layer. Keep the pinned
    # foundation's imported source bytes and archive identity pristine.
    directory = PORT / "backend/patches"
    return {str(path.relative_to(PORT)): sha(path)
            for path in sorted(directory.glob("*.patch"))}


def prepare_source(out, selected):
    root = out / "foundation"
    archive = out / "foundation.tar"
    if not archive.is_file() or sha(archive) != PIN["archive_sha256"]:
        if selected:
            repository = selected.resolve()
        else:
            repository = out / "source-git"
            if not (repository / "HEAD").is_file():
                run(["git", "init", "--bare", repository])
            run(["git", "-C", repository, "fetch", "--depth=1", PIN["repository"], PIN["revision"]])
        with archive.with_suffix(".part").open("wb") as output:
            run(["git", "-C", repository, "archive", "--format=tar", PIN["revision"], *PIN["paths"]], stdout=output)
        if sha(archive.with_suffix(".part")) != PIN["archive_sha256"]:
            raise RuntimeError("Foundation Git archive does not match the reviewed pin")
        archive.with_suffix(".part").replace(archive)
    # Keep mtimes on cache hits. Validate pristine files even when build outputs
    # share this root; never replace o/ or invalidate a completed LLVM build.
    with tarfile.open(archive) as tar:
        expected = set()
        for entry in tar.getmembers():
            if entry.isdir():
                continue
            if not entry.isfile():
                raise RuntimeError("Unexpected foundation archive member: " + entry.name)
            expected.add(entry.name)
            path = root / entry.name
            if not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("Foundation path escapes its output directory")
            content = tar.extractfile(entry).read()
            if not path.is_file() or path.read_bytes() != content:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod(entry.mode)
    # No stale source from a previous pin may join a shim or patch glob.
    for name in source_inventory(root):
        if name not in expected:
            (root / name).unlink()
    return root


def environment(out, sdk, jobs):
    tools = sdk.parent / "tools/bin"
    env = dict(os.environ)
    env.update(COSMOCC=str(sdk), COSMO=str(sdk), LLVM_COSMO_JOBS=str(jobs),
               LAVAPIPE_JOBS=str(jobs), CARGO_BUILD_JOBS=str(jobs))
    for name in ("LLVM_COSMO_OUT", "LAVAPIPE_OUT", "COSMO_RUST_BUILD_ROOT", "COSMO_RUST_ARCH"):
        env.pop(name, None)
    for key, name in (("CMAKE", "cmake"), ("NINJA", "ninja")):
        binary = tools / name
        if not binary.is_file():
            binary = Path(shutil.which(name) or "")
        if not binary.is_file():
            raise RuntimeError(name + " missing; run cosmopolitan/build.py --prepare-only first")
        env[key] = str(binary.resolve())
    env["PATH"] = str(tools) + os.pathsep + env.get("PATH", "")
    return env


def seed_sdk(root, sdk):
    archive = sdk.parent / "downloads/cosmocc-4.0.2.zip"
    if not archive.is_file() or sha(archive) != SDK_SHA:
        raise RuntimeError("Expected verified app Cosmocc archive alongside SDK")
    rust_sdk = root / "o/rust-ape/sdk"
    destination = rust_sdk / "vendor/cosmocc"
    if not destination.exists():
        # Same immutable SDK bytes, without a second download or disk copy.
        # A real directory is required by rust-ape's safe symlink restoration.
        try:
            shutil.copytree(sdk, destination, symlinks=True, copy_function=os.link)
        except OSError:
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(sdk, destination, symlinks=True)
    cached = rust_sdk / "cache/cosmocc-4.0.2.zip"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists():
        try:
            os.link(archive, cached)
        except OSError:
            shutil.copy2(archive, cached)
    write(rust_sdk / "vendor/.stamps/cosmocc", "cosmocc-4.0.2 sha256:" + SDK_SHA)
    return rust_sdk


def prepare_libclang(out, env):
    if env.get("LIBCLANG_PATH"):
        return
    tools = out / "host-tools"
    python = tools / "bin/python"
    library = tools / "libclang.path"
    if not library.is_file() or not Path(library.read_text().strip()).is_dir():
        if not python.is_file():
            run([sys.executable, "-m", "venv", tools])
        run([python, "-m", "pip", "install", "--disable-pip-version-check", "libclang==18.1.1"])
        path = subprocess.check_output([python, "-c", "import clang,pathlib; print(pathlib.Path(clang.__file__).parent/'native')"], text=True).strip()
        write(library, path + "\n")
    env["LIBCLANG_PATH"] = library.read_text().strip()


def build_rust(out, root, sdk, env):
    rust_sdk = seed_sdk(root, sdk)
    prepare_libclang(out, env)
    runner = root / "third_party/rust_ape/run.sh"
    run(["bash", runner, "setup"], cwd=root, env=env)
    run([sys.executable, root / "third_party/wgpu_native/prepare.py"], cwd=root, env=env)
    for name in application_patches():
        run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", PORT / name],
            cwd=root / "o/webgpu/source")
    env["WGPU_NATIVE_VERSION"] = "v" + PIN["wgpu_native_version"]
    env["BINDGEN_EXTRA_CLANG_ARGS"] = (f'-I"{sdk / "include"}" -include '
                                       f'"{sdk / "include/libc/normalize.inc"}"')
    webgpu = root / "o/webgpu"
    with (webgpu / "cargo-artifacts.jsonl").open("w") as log:
        run(["bash", runner, "cargo", "--config", webgpu / "cargo.cosmo.toml", "build",
             "--manifest-path", webgpu / "source/Cargo.toml", "--locked", "--release", "--lib",
             "--no-default-features", "--features", "wgsl,vulkan", "--message-format=json-render-diagnostics"],
            cwd=root, env=env, stdout=log)
    # The pinned linker compiles the Linux-personality runtime overrides that
    # Rust std uses. Link a build-only empty program to materialize its objects;
    # the application links these same objects directly before libcosmo.a.
    runtime = out / "runtime-link.c"
    write(runtime, "int main(void) { return 0; }\n")
    run([rust_sdk / "generated/linker-x86_64.bash", "-mcosmo", runtime,
         "-o", out / "runtime-link.com.dbg"], cwd=root, env=env)
    run([sys.executable, root / "third_party/wgpu_native/package_notices.py"], cwd=root, env=env)


def static_archive(path):
    with path.open("rb") as file:
        if file.read(8) != b"!<arch>\n":
            raise RuntimeError("Expected complete static archive: " + str(path))


def make_manifest(out, root, sdk):
    mesa_path = root / "o/lavapipe/LINK.json"
    mesa = json.loads(mesa_path.read_text())
    llvm = mesa["llvm"]
    if (mesa["mesa_version"] != PIN["mesa_version"] or
            llvm["llvm_version"] != PIN["llvm_version"] or
            not mesa["static"] or not llvm["static"] or
            mesa["external_vulkan_loader"] or Path(mesa["sdk_root"]).resolve() != sdk):
        raise RuntimeError("Software Vulkan metadata differs from pinned portable configuration")
    wgpu = root / "o/rust-ape/build/x86_64-unknown-linux-musl/release/libwgpu_native.a"
    shims = sorted((root / "o/rust-ape/sdk/generated").glob("shim-*-x86_64.o"))
    expected_shims = list((root / "third_party/rust_ape/upstream/shim").rglob("*.c"))
    if len(shims) != len(expected_shims):
        raise RuntimeError("Rust compatibility object set is incomplete")
    archives = [wgpu, Path(mesa["registration_archive"]), Path(mesa["archive"]), *map(Path, llvm["archives"])]
    for archive in archives:
        static_archive(archive)
    for record, field in ((mesa, "archive"), (mesa, "registration_archive"),
                          (mesa, "probe"), (llvm, "probe")):
        if sha(record[field]) != record[field + "_sha256"]:
            raise RuntimeError("Completed dependency metadata hash mismatch: " + record[field])
    if sha(mesa["llvm_metadata"]) != mesa["llvm_metadata_sha256"]:
        raise RuntimeError("Mesa was linked against different LLVM metadata")
    for arg in mesa["link_args"]:
        if any(x in arg.lower() for x in (".so", ".dll", ".dylib")):
            raise RuntimeError("Foreign shared library in software dependency closure: " + arg)
    notices = out / "licenses"
    shutil.copytree(root / "o/webgpu/licenses", notices / "webgpu", dirs_exist_ok=True)
    shutil.copytree(mesa["licenses"], notices / "software-vulkan", dirs_exist_ok=True)
    shutil.copy2(HERE / "PIN.json", notices / "SOFTWARE-WEBGPU-PIN.json")
    inputs = [*archives, *shims, mesa_path, Path(mesa["llvm_metadata"]),
              root / "o/webgpu/generated/vulkan_signatures.inc",
              root / "o/webgpu/source/Cargo.lock"]
    inputs.extend(sorted((root / "o/webgpu/source/src").rglob("*.rs")))
    manifest = {
        "format": 1, "pin": PIN, "recipe_sha256": sha(Path(__file__)),
        "application_patches": application_patches(),
        "foundation_root": str(root), "foundation_sources": source_inventory(root),
        "sdk_root": str(sdk), "sdk_archive_sha256": SDK_SHA,
        "target": "x86_64 Cosmopolitan APE", "static": True,
        "runtime_shared_libraries": [], "wgpu_archive": str(wgpu),
        "runtime_objects": list(map(str, shims)),
        "runtime_link_options": ["-Wl,--defsym,__isoc23_strtol=strtol", "-Wl,--defsym,__isoc23_sscanf=sscanf"],
        "include_dirs": [str(root / "third_party/wgpu_native/upstream/ffi"),
                         str(root / "third_party/wgpu_native/upstream/ffi/webgpu-headers"),
                         str(root / "third_party/wgpu_native/runtime"),
                         str(root / "o/webgpu/generated"), str(root / "third_party/lavapipe")],
        "runtime_sources": [str(root / "third_party/wgpu_native/runtime" / name)
                            for name in ("vulkan_loader.c", "win64_bridge.c")],
        "driver_link_args": mesa["link_args"], "software_vulkan": mesa,
        "file_sha256": {str(p): sha(p) for p in inputs},
        "licenses": str(notices),
        "license_sha256": {str(p.relative_to(notices)): sha(p) for p in sorted(notices.rglob("*")) if p.is_file()},
    }
    write(out / "LINK.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def cached(out, root, sdk):
    path = out / "LINK.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text())
    if (data.get("pin") != PIN or data.get("recipe_sha256") != sha(Path(__file__))
            or data.get("application_patches") != application_patches()
            or data.get("foundation_sources") != source_inventory(root)
            or data.get("sdk_root") != str(sdk)):
        return False
    for name, expected in data["file_sha256"].items():
        if not Path(name).is_file() or sha(name) != expected:
            return False
    for name, expected in data["license_sha256"].items():
        path = Path(data["licenses"]) / name
        if not path.is_file() or sha(path) != expected:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PORT / "out/software-webgpu")
    parser.add_argument("--sdk", type=Path, default=PORT / "out/sdk")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--foundation-source", type=Path, help="local Git object source for the exact pinned commit")
    parser.add_argument("--stage", choices=["all", "prepare", "rust", "llvm", "mesa", "manifest"], default="all")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    out, sdk = args.out.resolve(), args.sdk.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / (args.stage + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = prepare_source(out, args.foundation_source)
    env = environment(out, sdk, args.jobs)
    if args.stage == "prepare":
        print(root)
        return
    if args.stage == "all" and cached(out, root, sdk):
        print("Verified software WebGPU dependency cache: " + str(out / "LINK.json"))
        return
    if args.stage in ("all", "rust"):
        build_rust(out, root, sdk, env)
    if args.stage in ("all", "llvm"):
        run([sys.executable, root / "third_party/llvm_cosmo/build.py"], env=env, cwd=root)
    if args.stage in ("all", "mesa"):
        run([sys.executable, root / "third_party/lavapipe/build.py", "--llvm-metadata",
             root / "o/llvm-cosmo/LINK.json"], env=env, cwd=root)
    if args.stage in ("all", "manifest"):
        make_manifest(out, root, sdk)
        print("Static software WebGPU dependency manifest: " + str(out / "LINK.json"))


if __name__ == "__main__":
    main()
