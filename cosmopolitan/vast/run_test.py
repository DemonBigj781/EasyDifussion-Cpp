#!/usr/bin/env python3
"""Bounded tests of the published x86-64 APE payload; never downloads a model.

Preflight proves embedded and explicitly selected native software execution.
Hardware mode requires a real native discrete/integrated adapter before llama
or optional diffusion inference. Xavier/aarch64 needs a separate ARM artifact.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import subprocess
import sys
import time


EXPECTED_SOURCE = "1730424270e35ff4a06bd978654c9990cfd5777e"
EXPECTED_APP = "7dd513293f53bc39bd7c92e3aaff0522c89f0d543c3390b4dd95e2f590b507b9"
EXPECTED_BYTES = 190491826
DEFAULT_ARTIFACT = Path("/opt/edcpp-test/build/artifact")
COUNTERS = ("graphs", "submissions", "dispatches", "matmuls", "readbacks")
COUNTER_PATTERN = r"graphs=(\d+) submissions=(\d+) dispatches=(\d+) matmuls=(\d+) readbacks=(\d+)"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def check_host():
    system, machine = platform.system(), platform.machine().lower()
    require(system == "Linux" and machine in ("x86_64", "amd64"),
            "This pinned artifact requires Linux x86_64; " + system + "/" + machine +
            " is unsupported. Jetson Xavier/aarch64 requires a distinct ARM build.")
    return {"system": system, "machine": machine}


def checked_file(root, entry):
    relative = entry["path"]
    rel = PurePosixPath(relative)
    require(relative and not rel.is_absolute() and ".." not in rel.parts and "\\" not in relative,
            "Unsafe payload path")
    path = root.joinpath(*rel.parts)
    current = root
    for part in rel.parts:
        current /= part
        require(not current.is_symlink(), "Payload symlinks are not accepted: " + relative)
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), "Missing payload file: " + relative)
    require(type(entry["bytes"]) is int and entry["bytes"] > 0 and
            isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]),
            "Invalid payload identity: " + relative)
    actual = file_info(path)
    require(actual == {"bytes": entry["bytes"], "sha256": entry["sha256"]}, "Payload hash/size mismatch: " + relative)
    return path, {"path": str(path), **actual}


def load_parsers(artifact):
    """Called only after all parser files match the package pin."""
    loaded = {}
    sys.dont_write_bytecode = True
    for name in ("verify_runtime", "verify_webgpu_device", "verify_inference"):
        spec = importlib.util.spec_from_file_location(name, artifact / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def kill_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


class Commands:
    def __init__(self, output, environment, deadline, records):
        self.output, self.environment, self.deadline, self.records = output, environment, deadline, records

    def run(self, label, command, timeout, expected=0):
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, "Overall test deadline exhausted before " + label)
        timeout = min(timeout, remaining)
        record = {"label": label, "command": list(command), "expected_application_exit_code": expected,
                  "timeout_seconds": timeout, "stdout_log": label + ".stdout.log",
                  "stderr_log": label + ".stderr.log", "timed_out": False}
        self.records.append(record)
        started = time.monotonic()
        proc = None
        try:
            with (self.output / record["stdout_log"]).open("xb") as stdout, (self.output / record["stderr_log"]).open("xb") as stderr:
                proc = subprocess.Popen(command, cwd=self.output / "work", env=self.environment,
                                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                        start_new_session=True)
                try:
                    record["returncode"] = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
                    kill_group(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    finally:
                        # Kill descendants even if the parent already exited on SIGTERM.
                        kill_group(proc.pid, signal.SIGKILL)
                        proc.wait()
                        record["returncode"] = proc.returncode
                        record["cleanup"] = "owned process group SIGTERM then SIGKILL"
                    raise RuntimeError(label + " exceeded its bounded deadline")
            require(record["returncode"] == expected,
                    label + ": exit " + str(record["returncode"]) + ", expected " + str(expected))
            return ((self.output / record["stdout_log"]).read_text(errors="replace"),
                    (self.output / record["stderr_log"]).read_text(errors="replace"))
        finally:
            if proc is not None:
                kill_group(proc.pid, signal.SIGKILL)
                if proc.poll() is None:
                    proc.wait()
            record["elapsed_seconds"] = round(time.monotonic() - started, 6)


def validate_native_image(stdout, stderr, png, device, parser):
    """Native variant of the frozen embedded-only verifier's image contract."""
    lines, one = stdout.splitlines(), parser.unique_marker
    one(lines, "IMAGE_INFERENCE", r"IMAGE_INFERENCE PASS")
    selected = one(lines, "IMAGE_DEVICE", r"IMAGE_DEVICE selector=(\S+) provider=(native|embedded) software=([01]) stable_id=(\S+)")
    require(selected.groups()[:3] == (device["selector"], "native", str(int(device["software"]))),
            "Image selected a different provider/device/classification")
    stages = {}
    for stage in ("MODEL_LOAD", "GENERATION"):
        match = one(lines, "IMAGE_" + stage, r"IMAGE_" + stage + r" backend=webgpu success=1 seconds=([0-9.]+) " + COUNTER_PATTERN)
        seconds = float(match.group(1))
        require(math.isfinite(seconds) and seconds >= 0, "Invalid image stage duration")
        stages[stage.lower()] = {"seconds": seconds, **dict(zip(COUNTERS, map(int, match.groups()[1:])))}
    params = one(lines, "IMAGE_PARAMETERS", r"IMAGE_PARAMETERS width=256 height=256 steps=2 seed=42 threads=2 cfg_scale=7 sampler=(\S+) scheduler=(\S+) cpu_fallback_allowed=1")
    sampling = one(lines, "IMAGE_SAMPLING", r"IMAGE_SAMPLING backend=webgpu first_step=1 last_step=2 total_steps=2 completed_callbacks=2 intervals=1 monotonic=1 " + COUNTER_PATTERN)
    interval = dict(zip(COUNTERS, map(int, sampling.groups())))
    require(all(0 < interval[key] <= stages["generation"][key] for key in COUNTERS),
            "Missing positive generation and denoising-only native work")
    require(interval["matmuls"] <= interval["dispatches"] and
            stages["generation"]["matmuls"] <= stages["generation"]["dispatches"], "Invalid matmul counter")
    progress = []
    for line in stderr.splitlines():
        if not line.startswith("IMAGE_SAMPLE_PROGRESS"):
            continue
        match = re.fullmatch(r"IMAGE_SAMPLE_PROGRESS step=(\d+) total_steps=2 seconds=([0-9.]+)", line)
        require(match is not None, "Malformed sampling progress")
        seconds = float(match.group(2))
        require(math.isfinite(seconds) and seconds >= 0, "Invalid sampling duration")
        progress.append({"step": int(match.group(1)), "seconds": seconds})
    require([entry["step"] for entry in progress] == [1, 2], "Missing/duplicate/unordered completed sampling callbacks")
    executed = one(lines, "IMAGE_WEBGPU_EXECUTION", r"IMAGE_WEBGPU_EXECUTION provider=native software=([01]) native_loader_opens=(\d+) cpu_fallback_allowed=1 cpu_fallback_measured=0")
    require(int(executed.group(1)) == int(device["software"]) and int(executed.group(2)) > 0,
            "Image native-loader/device evidence differs")
    adapter = one(lines, "IMAGE_WEBGPU_ADAPTER", r"IMAGE_WEBGPU_ADAPTER (.+)").group(1)
    require(adapter == device["adapter_name"], "Image adapter name differs from selected graph adapter")
    output = one(lines, "IMAGE_OUTPUT", r"IMAGE_OUTPUT width=(\d+) height=(\d+) channels=(\d+) pixel_bytes=(\d+) png_bytes=(\d+) pixel_min=(\d+) pixel_max=(\d+) pixel_mean=([0-9.]+)")
    keys = ("width", "height", "channels", "pixel_bytes", "bytes", "pixel_min", "pixel_max")
    require(tuple(map(int, output.groups()[:7])) == tuple(png[key] for key in keys) and
            math.isclose(float(output.group(8)), png["pixel_mean"], abs_tol=1e-7), "Decoded PNG differs from application statistics")
    return {**stages, "provider": "native", "selector": device["selector"], "software": device["software"],
            "adapter": adapter, "native_loader_opens": int(executed.group(2)), "sampler": params.group(1),
            "scheduler": params.group(2), "cpu_fallback_allowed": True, "cpu_fallback_fraction_measured": False,
            "generation_scope": "Complete generate_image after context load, including lazy uploads, CLIP and VAE",
            "sampling": {**interval, "first_step": 1, "last_step": 2, "completed_callbacks": 2, "intervals": 1,
                         "progress": progress, "scope": "Between completed denoising steps; excludes first step, CLIP and VAE; previews disabled"}}


def device_commands(commands, base, provider, selector, require_hardware, timeout, parser):
    prefix = provider
    config = commands.output / "work" / (provider + "-config.json")
    common = ["--config", str(config), "--backend", "webgpu", "--provider", provider]
    flags = ["--require-hardware"] if require_hardware else []
    out, err = commands.run(prefix + "-graph", base + ["webgpu-device-test", *common, "--device", selector, *flags], timeout)
    result = parser.validate_graph(out, provider, require_hardware)
    lane = parser.one(out.splitlines(), "WEBGPU_DEVICE_LANE", r"WEBGPU_DEVICE_LANE mode=(direct|main-thread-service)").group(1)
    require(lane == ("direct" if provider == "embedded" else "main-thread-service"), "Wrong execution lane")
    if not require_hardware:
        require(result["software"] and result["adapter_type"] == 3, "Preflight requires an explicitly software CPU adapter")
    selected = [*common, "--device", result["selector"]]
    out, err = commands.run(prefix + "-llama", base + ["llama", *selected, "--prompt", "Once upon a time",
                           "--tokens", "16", "--threads", "1", "--report-tokens"], timeout)
    result["llama"] = parser.validate_llama(out, err, result)
    if provider == "embedded":
        require(result["llama"]["native_loader_opens"] == 0, "Embedded llama opened a native loader")
    if not require_hardware:
        out, err = commands.run(prefix + "-hardware-rejection", base + ["webgpu-device-test", *selected, "--require-hardware"], timeout, expected=1)
        expected = "WEBGPU_DEVICE_TEST FAIL: hardware required, selected " + result["selector"] + " provider=" + provider + " type=software-cpu"
        require(err.splitlines().count(expected) == 1 and "WEBGPU_DEVICE_EXECUTION" not in out and
                "WEBGPU_DEVICE_TEST PASS" not in out, "Software hardware-required refusal was not explicit")
        result["hardware_required_rejection_verified"] = True
    result["execution_lane"] = lane
    return result


def verify_bundle(artifact, manifest_path):
    root = manifest_path.parent
    metadata = json.loads(manifest_path.read_text())
    pin_path, pin_record = checked_file(root, metadata["package_pin"])
    require(pin_path == root / "PACKAGE_PIN.json", "Unexpected package pin location")
    pin = json.loads(pin_path.read_text())
    require(pin["schema"] == metadata["schema"] == 1 and
            pin["source_commit"] == metadata["source_commit"] == EXPECTED_SOURCE and
            pin["run_id"] == metadata["run_id"] == 37568302821, "Wrong package source/run/schema")
    require(pin["application"] == metadata["application"] and pin["loader"] == metadata["loader"] and
            pin["files"] == metadata["files"], "Payload differs from the checked-in package pin")
    require(pin["application"] == {"path": "artifact/easy-diffusion.exe", "bytes": EXPECTED_BYTES, "sha256": EXPECTED_APP},
            "Package does not identify the published application")
    expected = {"easy-diffusion.exe", "ape-x86_64.elf", "BUILD.json", "PIN.json",
                "verify_runtime.py", "verify_webgpu_device.py", "verify_inference.py"}
    entries = pin["files"]
    require(len(entries) == len(expected) and {x["path"] for x in entries} == {"artifact/" + x for x in expected},
            "Package pin must contain exactly the seven required files")
    before = {"package_pin": pin_record, "payload_manifest": {"path": str(manifest_path), **file_info(manifest_path)}}
    for entry in entries:
        local = dict(entry, path=entry["path"][len("artifact/"):])
        _, before[entry["path"]] = checked_file(artifact, local)
    require(next(x for x in entries if x["path"] == "artifact/easy-diffusion.exe") == pin["application"] and
            next(x for x in entries if x["path"] == "artifact/ape-x86_64.elf") == pin["loader"],
            "Duplicate application/loader identities disagree")
    build = json.loads((artifact / "BUILD.json").read_text())
    require(build["source"]["commit"] == EXPECTED_SOURCE and build["source"]["dirty"] is False and
            build["executable"]["sha256"] == EXPECTED_APP and build["executable"]["bytes"] == EXPECTED_BYTES,
            "Packaged BUILD identity differs")
    require(json.loads((artifact / "PIN.json").read_text()) == pin["model"], "Model pin differs from published package")
    require(re.fullmatch(r"ubuntu:24\.04@sha256:[0-9a-f]{64}", metadata["base_image"]), "Container base image is not digest-pinned")
    helper = metadata["native_helper"]
    require(helper["path"] == "native-tmp/.cosmo/dlopen-helper" and
            helper["source"]["path"] == "native-tmp/.cosmo/dlopen-helper.c", "Wrong native helper layout")
    for key, entry in (("native_helper", helper), ("native_helper_source", helper["source"])):
        path, record = checked_file(root, entry)
        require(type(entry["mtime_ns"]) is int and path.stat().st_mtime_ns == entry["mtime_ns"], "Native helper timestamp differs: " + key)
        before[key] = {**record, "mtime_ns": entry["mtime_ns"]}
    helper_path = root / helper["path"]
    require(helper["mtime_ns"] >= helper["source"]["mtime_ns"], "Native helper is older than its source; rebuilding is forbidden")
    with helper_path.open("rb") as stream:
        header = stream.read(20)
    require(header[:6] == b"\x7fELF\x02\x01" and header[18:20] == b"\x3e\x00", "Native helper is not x86-64 ELF")
    require(os.access(helper_path, os.X_OK) and os.access(artifact / "ape-x86_64.elf", os.X_OK), "Helper/APE loader is not executable")
    preparation_path, before["helper_preparation"] = checked_file(root, metadata["helper_preparation"])
    preparation = json.loads(preparation_path.read_text())
    require(preparation["status"] == "passed" and preparation["source_commit"] == EXPECTED_SOURCE and
            preparation["application_sha256_before"] == preparation["application_sha256_after"] == EXPECTED_APP and
            preparation["base_image"] == metadata["base_image"] and preparation["native_helper"] == helper,
            "Native helper build-time reuse attestation did not pass or differs")
    compiler_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    names = ("cc", "gcc", "g++", "c++", "clang", "clang++", "rustc", "cargo", "cmake", "ninja", "make")
    found = [name for name in names if shutil.which(name, path=compiler_path)]
    require(not found, "Runtime image retains compiler/build entrypoints: " + ", ".join(found))
    parsers = load_parsers(artifact)
    parsers["verify_runtime"].verify_package(artifact, EXPECTED_APP)
    return {"pin": pin, "metadata": metadata, "parsers": parsers, "identities_before": before,
            "native_tmp": helper_path.parent.parent,
            "compiler_check": {"searched_names": list(names), "searched_path": compiler_path, "found": found,
                               "scope": "Known compiler/build entrypoints absent; child PATH independently set to /no-tools"}}


def verify(args):
    output = args.output_dir.absolute()
    require(not output.is_symlink() and (not output.exists() or (output.is_dir() and not any(output.iterdir()))),
            "Use a fresh or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "work").mkdir()
    report = {"schema": 1, "status": "failed", "mode": args.mode, "commands": [], "devices": [],
              "hardware_verified": False, "model_executed": False, "quality_assessed": False,
              "timeout_seconds": args.timeout, "filesystem_isolated": False,
              "scope": "Published x86-64 APE in a container; software preflight is not physical GPU validation"}
    started, before = time.monotonic(), {}
    deadline = started + args.timeout
    try:
        report["host"] = check_host()
        artifact = args.artifact_dir.resolve()
        manifest = (args.payload_manifest or artifact.parent / "PAYLOAD.json").resolve()
        bundle = verify_bundle(artifact, manifest)
        before.update(bundle["identities_before"])
        report.update(identities_before=before, source_commit=EXPECTED_SOURCE, application_sha256=EXPECTED_APP,
                      compiler_check=bundle["compiler_check"],
                      package_pin_sha256=before["package_pin"]["sha256"], base_image=bundle["metadata"]["base_image"],
                      helper_provenance="Container build-output hash attestation; not a separately signed upstream artifact")
        if args.mode == "package":
            require(args.model is None and args.vulkan_icd is None, "Package-only mode accepts neither model nor ICD")
            report["scope"] = "Package/resource/helper integrity and known compiler absence only; no application, model or driver execution"
            report["status"] = "passed"
            return report
        if args.mode == "preflight":
            require(args.vulkan_icd is not None, "Preflight requires an explicit software --vulkan-icd")
        env = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "TERM": "dumb", "PATH": "/no-tools",
               "TMPDIR": str(bundle["native_tmp"]), "LP_NUM_THREADS": "2"}
        # Host GPU library search is the sole inherited native-loader setting.
        # LD_PRELOAD, compiler variables, credentials and arbitrary Vulkan layers
        # are deliberately not copied into the child environment.
        library_path = args.native_library_path if args.native_library_path is not None else os.environ.get("LD_LIBRARY_PATH")
        if library_path:
            env["LD_LIBRARY_PATH"] = library_path
        if args.vulkan_icd:
            icd = args.vulkan_icd.resolve()
            require(icd.is_file(), "Explicit Vulkan ICD JSON is missing")
            env.update(VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd))
            report["vulkan_icd"] = {"path": str(icd), **file_info(icd)}
        report["child_environment"] = {"PATH": "/no-tools", "TMPDIR": str(bundle["native_tmp"]),
                                       "LP_NUM_THREADS": "2", "native_loader_keys": sorted(k for k in env if k in
                                       ("LD_LIBRARY_PATH", "VK_DRIVER_FILES", "VK_ICD_FILENAMES"))}
        model = None
        if args.model:
            model = args.model.resolve()
            pin = bundle["pin"]["model"]
            require(model.is_file(), "Model file is missing; this runner never downloads models")
            before["model"] = {"path": str(model), **file_info(model)}
            require({k: before["model"][k] for k in ("bytes", "sha256")} ==
                    {k: pin[k] for k in ("bytes", "sha256")}, "Model differs from pinned SD1.5 checkpoint")
        commands = Commands(output, env, deadline, report["commands"])
        base = [str(artifact / "ape-x86_64.elf"), str(artifact / "easy-diffusion.exe")]
        parser = bundle["parsers"]["verify_webgpu_device"]
        if args.mode == "preflight":
            report["devices"].append(device_commands(commands, base, "embedded", "auto", False, args.command_timeout, parser))
        device = device_commands(commands, base, "native", args.device or ("auto" if args.mode == "hardware" else "WebGPU0"),
                                 args.mode == "hardware", args.command_timeout, parser)
        report["devices"].append(device)
        report["hardware_verified"] = device["hardware_verified"]
        if model is not None:
            require(args.mode != "hardware" or report["hardware_verified"], "Hardware gate did not pass before image inference")
            config = output / "work/native-config.json"
            out, err = commands.run("image", base + ["image", "--config", str(config), "--model", str(model),
                "--backend", "webgpu", "--provider", "native", "--device", device["selector"],
                "--prompt", "a red apple on a wooden table", "--output", str(output / "image.png"),
                "--width", "256", "--height", "256", "--steps", "2", "--threads", "2", "--seed", "42", "--cfg-scale", "7"],
                args.image_timeout)
            image_parser = bundle["parsers"]["verify_inference"]
            png = image_parser.inspect_png(output / "image.png", 256, 256)
            report["image"] = {"png": png, "execution": validate_native_image(out, err, png, device, image_parser),
                               "scope": "Pinned real-model two-step plumbing; no quality benchmark"}
            report["model_executed"] = True
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
    finally:
        after = {}
        for name, record in before.items():
            path = Path(record["path"])
            try:
                actual = {"path": str(path), **file_info(path)}
                if "mtime_ns" in record:
                    actual["mtime_ns"] = path.stat().st_mtime_ns
                after[name] = actual
                require(actual == record, "Verified file changed: " + name)
            except Exception as error:
                report.update(status="failed", error=str(error))
        report["identities_after"] = after
        report["identities_unchanged"] = bool(before) and before == after
        report["elapsed_seconds"] = round(time.monotonic() - started, 6)
        if time.monotonic() > deadline:
            report.update(status="failed", error="Overall test deadline exceeded")
        report["evidence"] = {p.name: file_info(p) for p in sorted(output.iterdir())
                              if p.is_file() and p.suffix in (".log", ".png")}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--payload-manifest", type=Path)
    parser.add_argument("--mode", choices=("package", "preflight", "hardware"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vulkan-icd", type=Path)
    parser.add_argument("--native-library-path")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--command-timeout", type=float, default=300)
    parser.add_argument("--image-timeout", type=float, default=1500)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 or x > 3600 for x in (args.timeout, args.command_timeout, args.image_timeout)):
        parser.error("Timeouts must be positive and no greater than 3600 seconds")
    if args.device is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", args.device):
        parser.error("Invalid device selector")
    try:
        report = verify(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"status": report["status"], "mode": args.mode,
                      "hardware_verified": report["hardware_verified"], "model_executed": report["model_executed"],
                      "report": str(args.output_dir / "report.json"), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
