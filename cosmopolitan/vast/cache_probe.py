#!/usr/bin/env python3
"""Measure sccache compatibility with the real Cosmopolitan C/C++/Rust recipes.

No application, LLVM or Mesa build is performed. Each cache claim requires a
second compilation at the same paths after removing outputs, with actual hit
counters. Unsupported compiler/target combinations remain disabled. This is a
Linux x86-64 CI probe, not a change to any production build's cache settings.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import time


HERE = Path(__file__).resolve().parent
PORT = HERE.parent
SCCACHE_VERSION = "0.16.0"
SCCACHE_ACTION = "fc920bf0ec8de6ee65d409111f7ec508035751ba"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count(value):
    """sccache's PerLanguageCount contains counts and overlapping adv_counts."""
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return sum(count(x) for x in value.get("counts", value).values())
    raise RuntimeError("Unrecognized sccache counter schema")


class Probe:
    def __init__(self, out, environment, timeout):
        self.out, self.env = out, environment
        self.deadline = time.monotonic() + timeout
        self.commands = []

    def run(self, name, args, *, cwd=None, env=None, required=True, timeout=900):
        args = [str(x) for x in args]
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Cache probe exhausted its overall deadline")
        stdout, stderr = self.out / (name + ".stdout.log"), self.out / (name + ".stderr.log")
        started = time.monotonic()
        item = {"name": name, "argv": args, "stdout": stdout.name, "stderr": stderr.name}
        self.commands.append(item)
        with stdout.open("xb") as so, stderr.open("xb") as se:
            child = subprocess.Popen(args, cwd=cwd, env=env or self.env,
                                     stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                                     start_new_session=True)
            try:
                item["returncode"] = child.wait(timeout=min(timeout, remaining))
            except subprocess.TimeoutExpired:
                item["timed_out"] = True
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                item["returncode"] = child.returncode
                raise RuntimeError("Timed out: " + name)
            finally:
                item["seconds"] = round(time.monotonic() - started, 6)
                item["stdout_sha256"], item["stderr_sha256"] = digest(stdout), digest(stderr)
        if required and item["returncode"]:
            raise RuntimeError(name + " failed; see " + stderr.name)
        return item, stdout.read_text(errors="replace"), stderr.read_text(errors="replace")

    def stats(self, name, sccache):
        _, stdout, _ = self.run(name, [sccache, "--show-stats", "--stats-format=json"], timeout=30)
        result = json.loads(stdout)
        stats = result.get("stats", result)
        if not isinstance(stats, dict) or "cache_hits" not in stats or "cache_misses" not in stats:
            raise RuntimeError("sccache stats lack explicit cache-hit/miss counters")
        return {"hits": count(stats["cache_hits"]), "misses": count(stats["cache_misses"]),
                "raw_stats": name + ".stdout.log"}


def c_probe(probe, build_recipe, sdk, wrappers, cmake, ninja, sccache):
    source, build = probe.out / "c-source", probe.out / "c-build"
    source.mkdir()
    (source / "answer.c").write_text("int c_answer(int n) { return n * 7; }\n")
    (source / "main.cpp").write_text(
        '#include <cstdio>\nextern "C" int c_answer(int);\n'
        'int main() { if (c_answer(6) != 42) return 1; '
        'std::puts("COSMO_CACHE_C_CPP PASS value=42"); return 0; }\n')
    (source / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.20)\nproject(cosmo_cache_probe LANGUAGES C CXX)\n'
        'add_library(cache_c OBJECT answer.c)\nadd_library(cache_cpp OBJECT main.cpp)\n'
        'add_executable(cache_probe $<TARGET_OBJECTS:cache_c> $<TARGET_OBJECTS:cache_cpp>)\n'
        'set_property(TARGET cache_probe PROPERTY LINKER_LANGUAGE CXX)\n'
        'set_property(TARGET cache_probe PROPERTY SUFFIX ".exe")\n')
    environment = dict(probe.env, COSMO_SDK=str(sdk), COSMO_BUILD_TOOLS=str(wrappers))
    configure = [cmake, "-S", source, "-B", build, "-G", "Ninja",
                 "-DCMAKE_MAKE_PROGRAM=" + str(ninja),
                 "-DCMAKE_TOOLCHAIN_FILE=" + str(PORT / "cmake/cosmocc.cmake"),
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DCMAKE_C_COMPILER_LAUNCHER=" + str(sccache),
                 "-DCMAKE_CXX_COMPILER_LAUNCHER=" + str(sccache)]
    probe.run("c-configure", configure, env=environment)
    result = {"status": "unsupported", "recommended_enabled": False,
              "same_paths_recompiled": False,
              "toolchain_file": str(PORT / "cmake/cosmocc.cmake"),
              "toolchain_sha256": digest(PORT / "cmake/cosmocc.cmake"),
              "wrappers": {name: {"path": str(wrappers / name), "sha256": digest(wrappers / name)}
                           for name in ("cc", "cxx")}, "launcher": str(sccache)}
    first, _, _ = probe.run("c-first-build", [cmake, "--build", build, "--target", "cache_probe", "--verbose"],
                            env=environment, required=False)
    if first["returncode"]:
        result["reason"] = "The actual C/C++ wrapper compilation failed through sccache; see c-first-build logs."
        # Confirm that a cache incompatibility is not misreported as a broken
        # Cosmopolitan compiler: the same tiny sources must still build/run.
        probe.run("c-disable-unsupported-cache", configure[:-2] +
                  ["-DCMAKE_C_COMPILER_LAUNCHER=", "-DCMAKE_CXX_COMPILER_LAUNCHER="], env=environment)
        probe.run("c-uncached-fallback", [cmake, "--build", build, "--target", "cache_probe"], env=environment)
    else:
        probe.run("c-clean", [cmake, "--build", build, "--target", "clean"], env=environment)
        result["same_paths_recompiled"] = True
        before = probe.stats("c-before-second", sccache)
        probe.run("c-second-c", [cmake, "--build", build, "--target", "cache_c", "--verbose"], env=environment)
        after_c = probe.stats("c-after-second-c", sccache)
        probe.run("c-second-cpp", [cmake, "--build", build, "--target", "cache_cpp", "--verbose"], env=environment)
        after_cpp = probe.stats("c-after-second-cpp", sccache)
        result.update(second_c_hits=after_c["hits"] - before["hits"],
                      second_cpp_hits=after_cpp["hits"] - after_c["hits"],
                      stats=[before, after_c, after_cpp])
        if result["second_c_hits"] > 0 and result["second_cpp_hits"] > 0:
            result.update(status="passed", recommended_enabled=True)
        else:
            result["reason"] = "At least one language produced no cache hit after an actual clean/recompile."
        probe.run("c-second-link", [cmake, "--build", build, "--target", "cache_probe"], env=environment)
    executable = build / "cache_probe.exe"
    with executable.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise RuntimeError("Cosmocc did not package the requested .exe as a PE-capable APE")
    original = digest(executable)
    _, stdout, _ = probe.run("c-ape-run", [sdk / "bin/ape-x86_64.elf", executable], timeout=30)
    if stdout.splitlines() != ["COSMO_CACHE_C_CPP PASS value=42"] or digest(executable) != original:
        raise RuntimeError("Linked C/C++ APE failed correctness or changed its executable")
    result.update(ape_sha256=original, ape_correctness="passed")
    result["compile_commands_sha256"] = digest(build / "compile_commands.json")
    return result


def rust_probe(probe, software_out, sdk, foundation_source, sccache):
    recipe = load("cosmo_cache_software", PORT / "software-webgpu/prepare.py")
    software_out.mkdir(parents=True, exist_ok=True)
    foundation = recipe.prepare_source(software_out, foundation_source)
    rust_sdk = recipe.seed_sdk(foundation, sdk)
    runner = foundation / "third_party/rust_ape/run.sh"
    environment = dict(probe.env, CARGO_INCREMENTAL="0", COSMO_RUST_ARCH="x86_64")
    for key in ("COSMO_RUST_BUILD_ROOT", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
        environment.pop(key, None)
    result = {"status": "unsupported", "recommended_enabled": False,
              "foundation_revision": recipe.PIN["revision"], "runner_sha256": digest(runner),
              "incremental": False, "crate_type": "rlib", "build_std": ["std", "panic_abort", "panic_unwind"],
              "runtime_cfgs_preserved_by": "pinned rust_ape/run.sh cargo", "stats": []}
    probe.run("rust-setup", ["bash", runner, "setup"], cwd=foundation, env=environment, timeout=1200)
    crate = probe.out / "rust-source"
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        '[package]\nname="cosmo_cache_probe"\nversion="0.1.0"\nedition="2021"\n'
        '[lib]\ncrate-type=["rlib"]\n[profile.release]\nopt-level=2\nincremental=false\n')
    (crate / "src/lib.rs").write_text(
        'pub fn answer(values: &[u32]) -> u32 {\n'
        '    let copied: Vec<u32> = values.iter().map(|x| x * 7).collect();\n'
        '    copied.iter().sum()\n}\n')
    cargo = ["bash", runner, "cargo"]
    build = [*cargo, "build", "--manifest-path", crate / "Cargo.toml", "--release", "--lib",
             "--message-format=json-render-diagnostics", "-j", "2"]
    # Warm the patched standard library without sccache. A failure here is a
    # toolchain/setup failure, not evidence that Rust caching is unsupported.
    probe.run("rust-uncached-baseline", build, cwd=crate, env=environment, timeout=1200)
    target = rust_sdk / "generated/x86_64-unknown-linux-musl.json"
    result.update(target=str(target), target_sha256=digest(target),
                  target_json=json.loads(target.read_text()), cargo_lock_sha256=digest(crate / "Cargo.lock"))
    toolchain = foundation / "o/rust-ape/toolchain"
    version_env = dict(environment, CARGO_HOME=str(toolchain / "cargo"), RUSTUP_HOME=str(toolchain / "rustup"),
                       RUSTUP_TOOLCHAIN=recipe.PIN["rust_toolchain"])
    _, rust_version, _ = probe.run("rust-version", [toolchain / "cargo/bin/rustc", "--version", "--verbose"], env=version_env)
    result["rustc_version"] = rust_version.strip()
    cached = dict(environment, RUSTC_WRAPPER=str(sccache))
    clean = [*cargo, "clean", "--manifest-path", crate / "Cargo.toml", "--release", "--package", "cosmo_cache_probe",
             "--target", target]
    snapshots = []
    for iteration in (1, 2):
        probe.run(f"rust-clean-{iteration}", clean, cwd=crate, env=environment, timeout=60)
        before = probe.stats(f"rust-before-{iteration}", sccache)
        item, stdout, _ = probe.run(f"rust-cached-{iteration}", [*build, "--locked"], cwd=crate,
                                    env=cached, required=False, timeout=600)
        after = probe.stats(f"rust-after-{iteration}", sccache)
        result["stats"].append({"before": before, "after": after, "hits": after["hits"] - before["hits"]})
        if item["returncode"]:
            result["reason"] = f"Pinned custom-target cargo compilation failed with sccache (attempt {iteration}); see rust-cached-{iteration} logs."
            return result
        artifacts = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
        own = [entry for entry in artifacts if entry.get("reason") == "compiler-artifact" and
               entry.get("target", {}).get("name") == "cosmo_cache_probe"]
        if len(own) != 1 or own[0].get("fresh") is not False:
            raise RuntimeError("Cargo did not actually recompile the cleaned custom-target probe")
        other_compilations = [entry.get("target", {}).get("name") for entry in artifacts
                              if entry.get("reason") == "compiler-artifact" and
                              entry.get("fresh") is False and entry not in own]
        if iteration == 2 and other_compilations:
            raise RuntimeError("Cannot attribute Rust replay hits to the probe: other crates recompiled: " +
                               ", ".join(str(x) for x in other_compilations))
        paths = [Path(name) for name in own[0]["filenames"] if name.endswith(".rlib")]
        if len(paths) != 1 or not paths[0].is_file():
            raise RuntimeError("Cargo did not emit the expected probe rlib")
        snapshots.append({"path": str(paths[0]), "sha256": digest(paths[0])})
    result["artifacts"] = snapshots
    if snapshots[0] != snapshots[1]:
        raise RuntimeError("Rust cache replay changed the same-path rlib output")
    if result["stats"][1]["hits"] > 0:
        result.update(status="passed", recommended_enabled=True)
    else:
        result["reason"] = "Pinned custom-target rlib compiled but yielded no cache hit after package clean."
    result["scope"] = "One std-using rlib via patched build-std/custom JSON target; not a Rust executable, WGPU build, or whole-application cache proof."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sdk-out", type=Path, default=PORT / "out")
    parser.add_argument("--software-out", type=Path, default=PORT / "out/software-webgpu")
    parser.add_argument("--foundation-source", type=Path)
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or args.timeout < 60 or args.timeout > 3600:
        parser.error("Use a new output directory and a timeout between 60 and 3600 seconds")
    out.mkdir(parents=True)
    report = {"schema": 1, "status": "failed", "scope": "Small compiler-cache compatibility experiment only; no production cache enabled",
              "sccache_action_commit": SCCACHE_ACTION, "sccache_required_version": SCCACHE_VERSION,
              "probe_sha256": digest(__file__), "c_cpp": {"status": "not_run"},
              "rust": {"status": "not_run", "recommended_enabled": False}}
    environment = dict(os.environ)
    # A private local server prevents other build jobs from contributing hits
    # or being stopped by this probe; configured remote cache storage is kept.
    with socket.socket() as temporary_socket:
        temporary_socket.bind(("127.0.0.1", 0))
        server_port = temporary_socket.getsockname()[1]
    environment.pop("SCCACHE_SERVER_UDS", None)
    environment.update(SCCACHE_SERVER_PORT=str(server_port), SCCACHE_DIR=str(out / "cache"), SCCACHE_IDLE_TIMEOUT="0")
    probe = Probe(out, environment, args.timeout)
    sccache = shutil.which("sccache")
    server_started = False
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
            raise RuntimeError("This cache probe requires Linux x86_64")
        if not sccache:
            raise RuntimeError("Pinned sccache is missing; install v0.16.0 with the pinned action first")
        _, version, _ = probe.run("sccache-version", [sccache, "--version"])
        if version.strip() != "sccache " + SCCACHE_VERSION:
            raise RuntimeError("Unexpected sccache version: " + version.strip())
        report["sccache_version"] = version.strip()
        probe.run("sccache-start", [sccache, "--start-server"], timeout=60)
        server_started = True
        recipe = load("cosmo_cache_build", PORT / "build.py")
        args.sdk_out = args.sdk_out.resolve()
        args.sdk_out.mkdir(parents=True, exist_ok=True)
        sdk = recipe.sdk_setup(args.sdk_out)
        cmake, ninja = recipe.host_tools(args.sdk_out)
        wrappers = recipe.make_wrappers(out, sdk)
        report["toolchain"] = {"cosmocc_version": recipe.SDK_VERSION, "archive_sha256": recipe.SDK_SHA256,
                               "build_recipe_sha256": digest(PORT / "build.py"), "sdk": str(sdk)}
        _, compiler, _ = probe.run("cosmocc-version", [wrappers / "cc", "--version"])
        report["toolchain"]["compiler_version"] = compiler.strip()
        report["c_cpp"] = c_probe(probe, recipe, sdk, wrappers, cmake, ninja, sccache)
        if args.skip_rust:
            report["rust"]["reason"] = "Explicit --skip-rust; no Rust cache compatibility claim"
        else:
            report["rust"] = rust_probe(probe, args.software_out.resolve(), sdk, args.foundation_source, sccache)
        report["status"] = "completed"
    except Exception as error:
        report["error"] = str(error)
    finally:
        if server_started:
            # Shutdown is bounded independently even after the probe deadline.
            try:
                subprocess.run([sccache, "--stop-server"], env=environment, timeout=20,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            except Exception as error:
                report["shutdown_error"] = str(error)
        report["commands"] = probe.commands
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "c_cpp": report["c_cpp"]["status"],
                      "rust": report["rust"]["status"], "report": str(out / "report.json"),
                      "error": report.get("error")}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
