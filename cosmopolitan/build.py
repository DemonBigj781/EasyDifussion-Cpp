#!/usr/bin/env python3
"""Build one x86-64 Cosmopolitan application from the vendored source trees."""

import argparse
from contextlib import contextmanager
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
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SDK_VERSION = "4.0.2"
SDK_URL = "https://cosmo.zip/pub/cosmocc/cosmocc-4.0.2.zip"
SDK_SHA256 = "85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44"
LLAMA_REVISION = "c589f0ed10c643678c4707dd160c21ac7633ebc0"
COMPILER_WRAPPER_REVISION = "preprocess-parity-v1"


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
    # Cosmocc 4.0.2 omits PREDEF/CFLAGS/PRECIOUS in its -E/-M/-MM branch.
    # sccache uses that branch before compilation, so it otherwise sees host
    # Linux macros and different integer typedefs. Derive the correction from
    # the pinned script rather than duplicating its language/mode flag logic.
    # The installed SDK remains unchanged; normal compilation still uses it.
    original = (sdk / "bin/cosmocross").read_text()
    old = 'if [ $INTENT = cpp ]; then\n  set -- "$CC" $PLATFORM $CPPFLAGS "$@"\n'
    new = ('if [ $INTENT = cpp ]; then\n'
           '  set -- "$CC" $PLATFORM $PREDEF $CFLAGS $CPPFLAGS "$@" $PRECIOUS\n')
    if original.count(old) != 1:
        raise RuntimeError("Pinned Cosmocc preprocessing branch changed; review wrapper parity")
    preprocess = directory / "cosmocross-preprocess.sh"
    write_if_changed(preprocess, original.replace(old, new))
    for name, tool in {
        "cc": "x86_64-unknown-cosmo-cc", "cxx": "x86_64-unknown-cosmo-c++",
        "ar": "x86_64-unknown-cosmo-ar", "ranlib": "x86_64-linux-cosmo-ranlib",
        "strip": "x86_64-unknown-cosmo-strip",
    }.items():
        binary = sdk / "bin" / tool
        content = ("#!" + sys.executable + "\nimport os,sys\n"
                   + "binary=" + repr(str(binary)) + "\n"
                   + "loader=" + repr(str(loader)) + "\n")
        if name in ("cc", "cxx"):
            content += ("# " + COMPILER_WRAPPER_REVISION + "\n"
                        + "preprocess=" + repr(str(preprocess)) + "\n"
                        + "intent=[x for x in sys.argv[1:] if x in ('-E','-M','-MM','-c','-S')]\n"
                        + "if intent and intent[-1] in ('-E','-M','-MM'):\n"
                        + "    with open(preprocess) as f: script=f.read()\n"
                        + "    os.execv('/bin/sh',['/bin/sh','-c',script,binary]+sys.argv[1:])\n")
        content += ("with open(binary,'rb') as f: magic=f.read(4)\n"
                   + "argv=([loader,binary] if magic[:2]==b'MZ' else [binary])+sys.argv[1:]\n"
                   + "os.execv(argv[0],argv)\n")
        write_if_changed(directory / name, content, executable=True)
    return directory


def application_source_entries(paths):
    """Enumerate exact staged paths; never treat a gitlink as an ordinary file.

    libwebm is deliberately excluded because the portable CMake recipe forces
    SD_WEBM=OFF. Keep its commit in the source fingerprint, and reject unknown
    gitlinks rather than silently dropping future runtime dependencies.
    """
    records = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z", "--", *paths], cwd=ROOT)
    names, omitted = [], []
    for record in records.split(b"\0"):
        if not record:
            continue
        metadata, raw_name = record.split(b"\t", 1)
        mode, revision, stage = metadata.decode().split()
        name = raw_name.decode()
        if stage != "0":
            raise RuntimeError("Unmerged application source: " + name)
        if mode == "160000":
            if name != "source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/libwebm":
                raise RuntimeError("Unreviewed application gitlink: " + name)
            omitted.append((name, revision))
        elif mode in ("100644", "100755", "120000"):
            names.append(name)
        else:
            raise RuntimeError("Unsupported application source mode: " + mode)
    return names, omitted


def prepare_source(out):
    paths = ["source/llama.cpp", "source/sdkit3-port-source", "source/UI.cpp"]
    names, omitted_gitlinks = application_source_entries(paths)
    identity = [(name, "disabled-gitlink", revision) for name, revision in omitted_gitlinks]
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
        # Construct and patch a fresh tree before replacing any active inputs.
        # An interrupted preparation cannot mix pristine and patched files.
        with tempfile.TemporaryDirectory(prefix="application-source-", dir=out) as staging:
            stage = Path(staging)
            for name in names:
                original = ROOT / name
                target = stage / Path(name).relative_to("source")
                target.parent.mkdir(parents=True, exist_ok=True)
                if original.is_symlink():
                    target.symlink_to(os.readlink(original))
                else:
                    shutil.copy2(original, target)
            for patch in patches:
                working = stage / "sdkit3-port-source" if patch.parent.name == "app-patches" else stage
                run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", patch], cwd=working)
            for path in paths:
                dest = source / Path(path).name
                if dest.exists():
                    shutil.rmtree(dest)
                (stage / Path(path).name).replace(dest)
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


SCCACHE_VERSION = "0.16.0"


def cache_counts(snapshot, key):
    """PerLanguageCount.adv_counts overlaps counts; do not sum both."""
    value = snapshot["stats"][key]["counts"]
    if not isinstance(value, dict) or any(type(n) is not int or n < 0 for n in value.values()):
        raise RuntimeError("Unrecognized sccache per-language counters")
    return value


def cache_delta(before, after):
    result = {}
    for key in ("cache_hits", "cache_misses"):
        start, end = cache_counts(before, key), cache_counts(after, key)
        result[key] = {name: end.get(name, 0) - start.get(name, 0)
                       for name in sorted(start.keys() | end.keys())}
        if any(n < 0 for n in result[key].values()):
            raise RuntimeError("sccache server counters reset during this build")
    return result


class BuildCache:
    """One application compiler launcher, with unmodified Cosmocc/Rust recipes."""
    def __init__(self, launcher, out, mode):
        self.env = dict(os.environ)
        self.launcher = None
        self.path = out / "build-cache" / (mode + ".json")
        self.report = {"schema": 1, "status": "running", "mode": mode,
                       "enabled": launcher != "none", "timings": [], "replay": []}
        self.started = time.monotonic()
        if launcher != "none":
            executable = shutil.which(launcher)
            if not executable:
                raise RuntimeError("Compiler cache launcher is not executable: " + launcher)
            self.launcher = str(Path(executable).resolve())
            version = subprocess.check_output([self.launcher, "--version"], text=True, timeout=30).strip()
            if version != "sccache " + SCCACHE_VERSION:
                raise RuntimeError("Supported compiler launcher is sccache " + SCCACHE_VERSION + "; got " + version)
            # A workspace wrapper nests inside RUSTC_WRAPPER; never stack them.
            if self.env.get("RUSTC_WORKSPACE_WRAPPER"):
                raise RuntimeError("Unset RUSTC_WORKSPACE_WRAPPER before enabling the compiler cache")
            for name in ("RUSTC_WRAPPER", "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER"):
                value = self.env.get(name)
                if value and str(Path(shutil.which(value) or value).resolve()) != self.launcher:
                    raise RuntimeError("Conflicting compiler launcher in " + name)
            base_buster = "cosmocc-" + SDK_VERSION + "-" + SDK_SHA256
            buster = base_buster + "-" + COMPILER_WRAPPER_REVISION
            if self.env.get("SCCACHE_C_CUSTOM_CACHE_BUSTER", buster) not in (base_buster, buster):
                raise RuntimeError("SCCACHE_C_CUSTOM_CACHE_BUSTER must identify the pinned SDK and wrapper revision")
            # Only application wrappers implement preprocessing parity. Do not
            # silently enable caching for direct SDK compilers in dependencies.
            for name in ("CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER"):
                if self.env.get(name):
                    raise RuntimeError("Unset " + name + "; use --compiler-launcher for the normalized application wrappers only")
            self.env.update(RUSTC_WRAPPER=self.launcher, CARGO_INCREMENTAL="0",
                            SCCACHE_C_CUSTOM_CACHE_BUSTER=buster)
            self.report["configuration"] = {"enabled": True, "launcher": self.launcher, "version": version,
                "launcher_sha256": digest(self.launcher), "sdk_sha256": SDK_SHA256,
                "cache_buster": buster, "wrapper_revision": COMPILER_WRAPPER_REVISION,
                "rust_incremental": False,
                "c_cpp": "CMake compiler launcher before the existing Cosmocc wrappers",
                "rust": "RUSTC_WRAPPER inherited by the pinned custom-target Rust runner",
                "excluded": ["dependency CMake launchers", "Mesa Meson compilation", "direct compiler calls", "linking", "packaging"]}
            self.report["before"] = self.snapshot()

    def snapshot(self):
        snapshot = json.loads(subprocess.check_output(
            [self.launcher, "--show-stats", "--stats-format=json"], env=self.env, text=True, timeout=30))
        for key in ("cache_hits", "cache_misses"):
            cache_counts(snapshot, key)
        return snapshot

    @contextmanager
    def phase(self, name):
        started = time.monotonic()
        item = {"phase": name, "status": "failed"}
        self.report["timings"].append(item)
        try:
            yield
            item["status"] = "passed"
        finally:
            item["seconds"] = round(time.monotonic() - started, 6)

    def replay(self, build, ninja, env):
        """Recompile two real application objects; never relink or rerun models."""
        commands = json.loads((build / "compile_commands.json").read_text())
        for filename, language in (("math_compat.c", "C"), ("config.cpp", "C++")):
            source = HERE / "src" / filename
            entries = [entry for entry in commands if (Path(entry["directory"]) / entry["file"]).resolve() == source.resolve()]
            if len(entries) != 1:
                raise RuntimeError("Expected exactly one application compile command for " + filename)
            entry = entries[0]
            argv = entry.get("arguments") or shlex.split(entry["command"])
            output = Path(entry["directory"]) / argv[argv.index("-o") + 1]
            output = output.resolve()
            if not output.is_relative_to(build.resolve()) or not output.is_file():
                raise RuntimeError("Missing or unsafe application object: " + str(output))
            original = digest(output)
            before = self.snapshot()
            started = time.monotonic()
            # Preserve the built object even if a cache/backend/compiler fails.
            with tempfile.TemporaryDirectory(prefix="cache-replay-", dir=build) as temporary:
                backup = Path(temporary) / "original.o"
                shutil.copy2(output, backup)
                try:
                    output.unlink()
                    run([ninja, "-C", build, "-j", "1", str(output.relative_to(build))], env=env)
                    after = self.snapshot()
                    delta = cache_delta(before, after)
                    same = digest(output) == original
                    record = {"source": str(source.relative_to(ROOT)), "language": language,
                        "object": str(output.relative_to(build)), "before_sha256": original,
                        "after_sha256": digest(output), "delta": delta,
                        "seconds": round(time.monotonic() - started, 6), "before": before, "after": after}
                    self.report["replay"].append(record)
                    if not same or delta["cache_hits"].get("C/C++", 0) < 1:
                        raise RuntimeError("Real " + language + " object replay needs an identical output and a cache hit")
                except BaseException:
                    shutil.copy2(backup, output)
                    raise

    def finish(self, error=None):
        self.report["status"] = "failed" if error else "passed"
        if error:
            self.report["error"] = str(error)
        try:
            if self.launcher:
                self.report["after"] = self.snapshot()
                self.report["delta"] = cache_delta(self.report["before"], self.report["after"])
        except Exception as stats_error:
            self.report["statistics_error"] = str(stats_error)
            if not error:
                self.report["status"] = "failed"
                raise
        finally:
            self.report["seconds"] = round(time.monotonic() - self.started, 6)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.report, indent=2, sort_keys=True) + "\n")
            print("Build/cache report: " + str(self.path), flush=True)


def source_selection(args):
    from foundation_consumer import select_foundation
    pin = json.loads((HERE / "software-webgpu/PIN.json").read_text())
    patches = {str(path.relative_to(HERE)): digest(path)
               for path in sorted((HERE / "backend/patches").glob("*.patch"))}
    return select_foundation(args.foundation_handoff, args.foundation_root, args.runtime_sdk,
                             args.software_webgpu_metadata, pin=pin, application_patches=patches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("COSMO_BUILD_JOBS", "2")))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--libraries-only", action="store_true")
    parser.add_argument("--dependencies-only", action="store_true",
                        help="complete the static software WebGPU closure before compiling application targets")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--software-webgpu-out", type=Path,
                        help="cache for the pinned static WGPU/Mesa/LLVM dependencies")
    parser.add_argument("--foundation-source", type=Path,
                        help="local Git object source for the exact pinned software WebGPU commit")
    parser.add_argument("--foundation-handoff", type=Path,
                        help="verified complete source-runtime handoff; requires the three export path options")
    parser.add_argument("--foundation-root", type=Path, help="materialized source foundation export root")
    parser.add_argument("--runtime-sdk", type=Path, help="complete source SDK inside the verified export")
    parser.add_argument("--software-webgpu-metadata", type=Path,
                        help="prebuilt application-patched source-runtime LINK.json inside the export")
    parser.add_argument("--compiler-launcher", default=os.environ.get("COSMO_COMPILER_LAUNCHER", "none"),
                        help="none (default), sccache, or its executable path; supported version 0.16.0")
    parser.add_argument("--cache-replay", action="store_true",
                        help="verify cache hits by rebuilding one actual C and C++ object after the application compile")
    args = parser.parse_args()
    if args.cache_replay and (args.compiler_launcher == "none" or args.prepare_only or args.dependencies_only or args.libraries_only):
        parser.error("--cache-replay requires a cached full application build")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source_options = (args.foundation_handoff, args.foundation_root, args.runtime_sdk,
                      args.software_webgpu_metadata)
    args.source_selection = None
    if any(source_options):
        if not all(source_options):
            parser.error("source runtime requires --foundation-handoff, --foundation-root, --runtime-sdk and --software-webgpu-metadata")
        if args.foundation_source or args.software_webgpu_out:
            parser.error("a prebuilt source handoff cannot be combined with legacy dependency staging options")
        if args.compiler_launcher != "none" or args.cache_replay:
            parser.error("source-runtime compiler caching requires separate preprocessing qualification; use --compiler-launcher none")
        # Keep the committed pin authoritative. This option cannot silently
        # upgrade it or activate an unreviewed candidate foundation revision.
        args.source_selection = source_selection(args)
    elif os.environ.get("COSMO_RUNTIME_PROFILE") == "source-built" or os.environ.get("COSMO_RUNTIME_SDK"):
        parser.error("source-runtime environment overrides require an explicit complete foundation handoff")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mode = "prepare" if args.prepare_only else "dependencies" if args.dependencies_only else "application"
        cache = BuildCache(args.compiler_launcher, out, mode)
        try:
            build_application(args, out, cache)
        except BaseException as error:
            cache.finish(error)
            raise
        else:
            cache.finish()


def build_application(args, out, cache):
    selected = args.source_selection
    if selected and args.dependencies_only:
        print("Verified prebuilt source-runtime dependencies: " + str(selected["link_manifest"]))
        return
    with cache.phase("prepare_sdk_and_tools"):
        sdk = selected["sdk"] if selected else sdk_setup(out)
        cmake, ninja = host_tools(out)
        if selected:
            from foundation_consumer import write_wrappers
            identity = selected["selection_identity_sha256"]
            wrappers = write_wrappers(out / ("toolchain-source-" + identity), selected)
            cache.report["configuration"] = {
                "enabled": False, "wrapper_revision": "source-direct-v1",
                "source_runtime": selected["runtime"], "handoff": selected["handoff_report"],
                "wrapper_sha256": {name: digest(wrappers / name) for name in ("cc", "cxx", "ar", "ranlib", "strip")},
            }
        else:
            wrappers = make_wrappers(out, sdk)
        if cache.launcher and not selected:
            cache.report["configuration"]["preprocessing_shim"] = {
                "sdk_driver_sha256": digest(sdk / "bin/cosmocross"),
                "shim_sha256": digest(wrappers / "cosmocross-preprocess.sh"),
                "cc_wrapper_sha256": digest(wrappers / "cc"),
                "cxx_wrapper_sha256": digest(wrappers / "cxx"),
            }
    with cache.phase("prepare_application_source"):
        source, shared, fingerprint = prepare_source(out)
    loader = selected["tools"]["ape_loader"] if selected else sdk / "bin/ape-x86_64.elf"
    shutil.copy2(loader, out / "ape-x86_64.elf")
    software_out = (args.software_webgpu_out or out / "software-webgpu").resolve()
    software_command = [sys.executable, HERE / "software-webgpu/prepare.py",
                        "--out", software_out, "--sdk", sdk, "--jobs", args.jobs]
    if args.foundation_source:
        software_command.extend(["--foundation-source", args.foundation_source.resolve()])
    if args.prepare_only:
        if not selected:
            run([*software_command, "--stage", "prepare"], env=cache.env)
        print("Prepared source: " + str(source))
        return
    # A missing/failed backend is a build failure; never emit a CPU-only
    # application under the all-in-one backend build's name.
    if selected:
        from foundation_consumer import write_link_view
        software_metadata = write_link_view(out / ("foundation-source-" + identity), selected)
    else:
        with cache.phase("software_webgpu_dependencies"):
            run(software_command, env=cache.env)
        software_metadata = software_out / "LINK.json"
    if args.dependencies_only:
        print("Prepared static software WebGPU dependencies: " + str(software_metadata))
        return
    if selected:
        # Do not let ambient include/library/compiler overrides turn a verified
        # source SDK into a mixed build. Host utilities still come from PATH.
        env = {key: cache.env[key] for key in
               ("HOME", "PATH", "TMPDIR", "TEMP", "TMP", "USER", "LOGNAME", "SYSTEMROOT")
               if key in cache.env}
        env.update(LC_ALL="C", COSMO_RUNTIME_PROFILE="source-built", COSMO_RUNTIME_SDK=str(sdk))
    else:
        env = dict(cache.env)
    env.update(COSMO_SDK=str(sdk), COSMO_BUILD_TOOLS=str(wrappers))
    env["PATH"] = str(wrappers) + os.pathsep + str(out / "tools/bin") + os.pathsep + env["PATH"]
    build = out / ("build-source-" + identity) if selected else out / "build"
    with cache.phase("configure_application"):
        run([cmake, "-S", HERE, "-B", build, "-G", "Ninja",
         "-DCMAKE_TOOLCHAIN_FILE=" + str(HERE / "cmake/cosmocc.cmake"),
         "-DCMAKE_MAKE_PROGRAM=" + str(ninja), "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
         "-DCMAKE_C_COMPILER_LAUNCHER=" + (cache.launcher or ""),
         "-DCMAKE_CXX_COMPILER_LAUNCHER=" + (cache.launcher or ""),
         "-DCOSMO_SOURCE_ROOT=" + str(source), "-DCOSMO_SHARED_GGML=" + str(shared),
         "-DCOSMO_SOFTWARE_WEBGPU_METADATA=" + str(software_metadata),
         "-DCOSMO_FOUNDATION_LINK_SHA256=" + (selected["link_view_sha256"] if selected else ""),
         "-DCOSMO_BUILD_APP=" + ("OFF" if args.libraries_only else "ON")], env=env)
    targets = ["llama", "stable-diffusion", "easy-diffusion-ui"] if args.libraries_only else ["easy-diffusion"]
    with cache.phase("compile_application"):
        run([cmake, "--build", build, "--parallel", args.jobs, "--target", *targets], env=env)
    if args.cache_replay:
        with cache.phase("replay_real_application_objects"):
            cache.replay(build, ninja, env)
    if selected:
        if source_selection(args) != selected:
            raise RuntimeError("Source foundation changed during application compilation")
        if software_metadata.is_symlink() or digest(software_metadata) != selected["link_view_sha256"]:
            raise RuntimeError("Generated foundation view changed during application compilation")
    if args.libraries_only:
        print("Shared GGML, llama.cpp, stable-diffusion.cpp, and UI static targets built")
        return
    debug = build / "bin/easy-diffusion.com.dbg"
    linked = out / "easy-diffusion.unpacked.exe"
    if selected:
        from foundation_consumer import tool_command
        package_command = tool_command(selected["tools"]["ape_linker"], loader)
    else:
        package_command = [loader, sdk / "bin/apelink"]
    run([*package_command, "-o", linked, "-l", loader, debug])
    set_stack(linked)
    shutil.copy2(debug, out / "easy-diffusion.com.dbg")
    metadata = {"cosmocc": {"version": SDK_VERSION, "url": SDK_URL, "sha256": SDK_SHA256},
                "compiler_cache": cache.report.get("configuration", {"enabled": False}),
                "llama_revision": LLAMA_REVISION, "source_fingerprint": fingerprint,
                "cpu_baseline": "x86-64; no AVX requirement", "shared_ggml": True,
                "openmp": False, "dynamic_backends": False,
                "vocabulary_translation_units": 12,
                "software_webgpu": json.loads(software_metadata.read_text()),
                "software_webgpu_metadata_sha256": digest(software_metadata)}
    if selected:
        metadata["cosmocc"] = {"version": "source-built", **selected["runtime"]}
        metadata["foundation_handoff"] = {
            **selected["handoff_report"], "manifest_sha256": selected["handoff_sha256"],
            "source_link_sha256": selected["link_sha256"],
            "generated_link_sha256": selected["link_view_sha256"],
            "export_root": str(selected["export_root"]),
            "selection_identity_sha256": selected["selection_identity_sha256"],
            "toolchain_selected": True, "application_compiled": True,
            "build_path": "explicit-prebuilt-source-runtime",
        }
    recipe_files = [HERE / "build.py", HERE / "build.sh", HERE / "CMakeLists.txt"]
    recipe_files.extend(HERE / name for name in ("foundation_handoff.py", "foundation_consumer.py"))
    for directory in ("cmake", "patches", "app-patches", "software-webgpu", "backend", "src"):
        recipe_files.extend(p for p in (HERE / directory).rglob("*") if p.is_file())
    metadata["build_recipe_sha256"] = {
        str(p.relative_to(HERE)): digest(p) for p in sorted(recipe_files)}
    shared_metadata = shared / "COSMOPOLITAN_SHARED_GGML.json"
    if shared_metadata.is_file():
        metadata["shared_ggml_metadata"] = json.loads(shared_metadata.read_text())
    manifest = out / "LINK.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if not args.skip_package:
        with cache.phase("package_application"):
            run([sys.executable, HERE / "package.py", "--input", linked,
                 "--output", out / "easy-diffusion.exe", "--metadata", manifest])
    print("Linked application: " + str(linked))


if __name__ == "__main__":
    main()
