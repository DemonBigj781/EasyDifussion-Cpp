#!/usr/bin/env python3
"""Verify real pinned SD1.5 inference through the packaged application's CLI.

Default 256x256/four-step inference is a plumbing check, not an image-quality
benchmark. The PNG is decoded independently using only the host Python standard
library. WebGPU counters cover both generation after model loading and the
interval between completed denoising steps (excluding the first step, CLIP and
VAE). CPU fallback is allowed and its fraction is not measured.
"""

import argparse
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import tempfile
import time
import zlib

from verify_runtime import APPLICATION, LOADER, runtime_environment, sha256, verify_package


COUNTERS = ("graphs", "submissions", "dispatches", "matmuls", "readbacks")


def read_model_pin(args):
    candidates = ([args.model_pin] if args.model_pin is not None else
                  [args.artifact_dir / "PIN.json", Path(__file__).resolve().parent / "PIN.json",
                   Path(__file__).resolve().parent.parent / "inference" / "PIN.json"])
    path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError("Model PIN.json is missing; supply --model-pin or package it beside the verifier")
    contents = path.read_bytes()
    pin = json.loads(contents)
    if not isinstance(pin, dict) or type(pin.get("schema")) is not int or pin["schema"] != 1:
        raise RuntimeError("Unsupported model pin schema")
    for field in ("repository", "revision", "filename", "sha256"):
        if not isinstance(pin.get(field), str) or not pin[field]:
            raise RuntimeError(f"Model pin requires a nonempty {field}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", pin["repository"]):
        raise RuntimeError("Model pin repository must identify an owner and repository")
    if not re.fullmatch(r"[0-9a-f]{40}", pin["revision"]) or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]):
        raise RuntimeError("Model pin requires exact lowercase revision and SHA256 values")
    if type(pin.get("bytes")) is not int or pin["bytes"] <= 0:
        raise RuntimeError("Model pin requires a positive integer byte size")
    if pin["filename"] in (".", "..") or any(char in pin["filename"] for char in ("/", "\\", "\0")):
        raise RuntimeError("Model pin filename must be a single filename")
    return pin, {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest(), "metadata": pin}


def paeth(a, b, c):
    p = a + b - c
    da, db, dc = abs(p - a), abs(p - b), abs(p - c)
    return a if da <= db and da <= dc else b if db <= dc else c


def inspect_png(path, width, height):
    """Decode the 8-bit, noninterlaced PNG emitted by the C image command.

    Bound decompression by the requested dimensions, validate every chunk CRC,
    and test distinct color pixels rather than just min/max across RGB channels
    (a solid red image has different channel extrema but is still constant).
    """
    if path.stat().st_size > 64 * 1024 * 1024:
        raise RuntimeError("Output exceeds the verification PNG size limit")
    data = path.read_bytes()
    if len(data) > 64 * 1024 * 1024 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("Output is not a bounded PNG file")
    cursor, header, ended = 8, None, False
    compressed = bytearray()
    chunks = []
    idat_closed = False
    while cursor < len(data):
        if cursor + 12 > len(data):
            raise RuntimeError("Truncated PNG chunk")
        length = struct.unpack_from(">I", data, cursor)[0]
        kind = data[cursor + 4:cursor + 8]
        end = cursor + 12 + length
        if end > len(data) or not re.fullmatch(b"[A-Za-z]{4}", kind):
            raise RuntimeError("Invalid PNG chunk bounds/type")
        payload = data[cursor + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if binascii.crc32(kind + payload) & 0xffffffff != crc:
            raise RuntimeError("PNG chunk CRC mismatch")
        if not chunks and kind != b"IHDR":
            raise RuntimeError("PNG does not start with IHDR")
        if kind == b"IHDR":
            if header is not None or length != 13:
                raise RuntimeError("Duplicate or invalid PNG IHDR")
            header = struct.unpack(">IIBBBBB", payload)
            w, h, bits, color, compression, filtering, interlace = header
            if (w, h) != (width, height):
                raise RuntimeError("PNG dimensions differ from the requested output")
            if bits != 8 or color not in (0, 2, 4, 6) or compression or filtering or interlace:
                raise RuntimeError("Expected an 8-bit noninterlaced grayscale/RGB PNG")
        elif kind == b"IDAT":
            if idat_closed:
                raise RuntimeError("Nonconsecutive PNG IDAT chunks")
            compressed.extend(payload)
        elif kind == b"IEND":
            if length or not compressed or end != len(data):
                raise RuntimeError("Invalid PNG IEND or trailing bytes")
            ended = True
        elif kind[0] & 32 == 0 and kind != b"PLTE":
            raise RuntimeError("Unknown critical PNG chunk")
        if compressed and kind != b"IDAT":
            idat_closed = True
        chunks.append(kind.decode("ascii"))
        cursor = end
    if not ended or header is None:
        raise RuntimeError("Incomplete PNG")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[header[3]]
    stride = width * channels
    expected = height * (stride + 1)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(bytes(compressed), expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise RuntimeError("PNG compressed pixel length/stream is invalid")
    previous = bytearray(stride)
    color_channels = 1 if channels in (1, 2) else 3
    colors = set()
    count, minimum, maximum, total, squared = 0, 255, 0, 0, 0
    rgba_minimum, rgba_maximum, rgba_total = 255, 0, 0
    pixel_hash = hashlib.sha256()
    for y in range(height):
        offset = y * (stride + 1)
        filter_type = raw[offset]
        if filter_type > 4:
            raise RuntimeError("Invalid PNG scanline filter")
        row = bytearray(raw[offset + 1:offset + 1 + stride])
        for x in range(stride):
            a = row[x - channels] if x >= channels else 0
            b = previous[x]
            c = previous[x - channels] if x >= channels else 0
            predictor = (0, a, b, (a + b) // 2, paeth(a, b, c))[filter_type]
            row[x] = (row[x] + predictor) & 255
        pixel_hash.update(row)
        rgba_minimum = min(rgba_minimum, min(row))
        rgba_maximum = max(rgba_maximum, max(row))
        rgba_total += sum(row)
        for x in range(0, stride, channels):
            pixel = bytes(row[x:x + color_channels])
            if len(colors) < 257:
                colors.add(pixel)
            for value in pixel:
                minimum, maximum = min(minimum, value), max(maximum, value)
                total += value
                squared += value * value
                count += 1
        previous = row
    if len(colors) < 2:
        raise RuntimeError("Decoded image has constant color pixels")
    mean = total / count
    return {"width": width, "height": height, "channels": channels,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "decoded_pixel_sha256": pixel_hash.hexdigest(), "chunk_crc_verified": True,
            "nonconstant_color_pixels": True, "distinct_colors_capped_at_257": len(colors),
            "color_min": minimum, "color_max": maximum, "color_mean": mean,
            "color_stddev": math.sqrt(max(0, squared / count - mean * mean)),
            "pixel_bytes": width * height * channels, "pixel_min": rgba_minimum,
            "pixel_max": rgba_maximum, "pixel_mean": rgba_total / (width * height * channels),
            "quality_assessed": False}


def unique_marker(lines, prefix, pattern):
    selected = [line for line in lines if line.startswith(prefix)]
    match = re.fullmatch(pattern, selected[0]) if len(selected) == 1 else None
    if not match:
        raise RuntimeError(f"Expected one valid {prefix} record")
    return match


def verify_markers(text, stderr_text, args, png):
    lines = text.splitlines()
    unique_marker(lines, "IMAGE_INFERENCE", r"IMAGE_INFERENCE PASS")
    stages = {}
    for stage in ("MODEL_LOAD", "GENERATION"):
        match = unique_marker(lines, "IMAGE_" + stage,
                              r"IMAGE_" + stage + r" backend=(cpu|webgpu) success=1 seconds=([0-9.]+) "
                              r"graphs=(\d+) submissions=(\d+) dispatches=(\d+) matmuls=(\d+) readbacks=(\d+)")
        seconds = float(match.group(2))
        if match.group(1) != args.backend or not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError("Invalid inference stage backend/timing")
        counters = dict(zip(COUNTERS, map(int, match.groups()[2:])))
        if counters["matmuls"] > counters["dispatches"]:
            raise RuntimeError("Matmul count exceeds dispatch count")
        stages[stage.lower()] = {"seconds": seconds, **counters}
    params = unique_marker(lines, "IMAGE_PARAMETERS",
                           r"IMAGE_PARAMETERS width=(\d+) height=(\d+) steps=(\d+) seed=(\d+) "
                           r"threads=(\d+) cfg_scale=([0-9.eE+-]+) sampler=(\S+) scheduler=(\S+) "
                           r"cpu_fallback_allowed=([01])")
    observed = tuple(map(int, params.groups()[:5]))
    if observed != (args.width, args.height, args.steps, args.seed, args.threads):
        raise RuntimeError("Application inference parameters differ from the request")
    cfg = float(params.group(6))
    if not math.isfinite(cfg) or not math.isclose(cfg, args.cfg_scale, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Application guidance scale differs from request")
    if int(params.group(9)) != int(args.backend == "webgpu"):
        raise RuntimeError("Unexpected diffusion CPU-fallback policy")
    for requested, reported in ((args.sampler, params.group(7)), (args.scheduler, params.group(8))):
        if requested is not None and requested != reported:
            raise RuntimeError("Application sampler/scheduler differs from request")
    sampling = unique_marker(lines, "IMAGE_SAMPLING",
                             r"IMAGE_SAMPLING backend=(cpu|webgpu) first_step=(\d+) last_step=(\d+) "
                             r"total_steps=(\d+) completed_callbacks=(\d+) intervals=(\d+) monotonic=1 "
                             r"graphs=(\d+) submissions=(\d+) dispatches=(\d+) matmuls=(\d+) readbacks=(\d+)")
    if sampling.group(1) != args.backend or tuple(map(int, sampling.groups()[1:6])) != (
            1, args.steps, args.steps, args.steps, args.steps - 1):
        raise RuntimeError("Sampling interval did not complete the requested ordered denoising steps")
    interval = dict(zip(COUNTERS, map(int, sampling.groups()[6:])))
    if interval["matmuls"] > interval["dispatches"] or any(
            interval[name] > stages["generation"][name] for name in COUNTERS):
        raise RuntimeError("Sampling counters exceed their enclosing generation/dispatch counts")
    progress = []
    for line in stderr_text.splitlines():
        if not line.startswith("IMAGE_SAMPLE_PROGRESS"):
            continue
        match = re.fullmatch(r"IMAGE_SAMPLE_PROGRESS step=(\d+) total_steps=(\d+) seconds=([0-9.]+)", line)
        if not match or int(match.group(2)) != args.steps:
            raise RuntimeError("Invalid completed-sampling progress record")
        seconds = float(match.group(3))
        if not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError("Invalid sampling duration")
        progress.append({"step": int(match.group(1)), "seconds": seconds})
    if [entry["step"] for entry in progress] != list(range(1, args.steps + 1)):
        raise RuntimeError("Missing, duplicate or nonmonotonic completed-sampling progress")
    output = unique_marker(lines, "IMAGE_OUTPUT",
                           r"IMAGE_OUTPUT width=(\d+) height=(\d+) channels=(\d+) pixel_bytes=(\d+) "
                           r"png_bytes=(\d+) pixel_min=(\d+) pixel_max=(\d+) pixel_mean=([0-9.]+)")
    actual = tuple(map(int, output.groups()[:7]))
    expected = (png["width"], png["height"], png["channels"], png["pixel_bytes"],
                png["bytes"], png["pixel_min"], png["pixel_max"])
    if actual != expected or not math.isclose(float(output.group(8)), png["pixel_mean"], abs_tol=1e-7):
        raise RuntimeError("Independent decoded pixels differ from application output statistics")
    result = {**stages, "sampler": params.group(7), "scheduler": params.group(8),
              "counter_scope": "Complete generate_image call after model/context loading; not UNet-only",
              "sampling": {"first_step": 1, "last_step": args.steps, "total_steps": args.steps,
                           "completed_callbacks": args.steps, "intervals": args.steps - 1,
                           "monotonic": True, **interval, "progress": progress,
                           "scope": "Denoising intervals between completed steps; excludes first step, CLIP and VAE; previews disabled"},
              "cpu_fallback_allowed": args.backend == "webgpu", "cpu_fallback_fraction_measured": False}
    if args.backend == "webgpu":
        unique_marker(lines, "IMAGE_WEBGPU_EXECUTION",
                      r"IMAGE_WEBGPU_EXECUTION provider=embedded software=1 native_loader_opens=0 "
                      r"cpu_fallback_allowed=1 cpu_fallback_measured=0")
        adapter = unique_marker(lines, "IMAGE_WEBGPU_ADAPTER", r"IMAGE_WEBGPU_ADAPTER (.+)").group(1)
        if any(stages["generation"][name] <= 0 for name in COUNTERS):
            raise RuntimeError("No actual post-model-load WebGPU inference/readback evidence")
        if any(interval[name] <= 0 for name in COUNTERS):
            raise RuntimeError("No actual WebGPU work in the denoising-only interval")
        result.update({"provider": "embedded", "software_adapter": True, "adapter": adapter,
                       "native_loader_opens": 0})
    elif any(line.startswith("IMAGE_WEBGPU_") for line in lines) or any(interval.values()) or any(
            stages[stage][name] for stage in stages for name in COUNTERS):
        raise RuntimeError("CPU-only inference unexpectedly reported WebGPU work")
    return result


class ProcessMemory:
    """Sample only a validated application process, across mounted PID namespaces.

    The workspace can expose an ancestor namespace's /proc: Popen.pid=6 may
    actually be /proc/67051 (NSpid: 67051 6), while /proc/6 is unrelated.
    Match the child namespace and its real parent, then validate its command,
    executable and executable application mapping before admitting any RSS.
    This measures the one application process, not an aggregate process tree.
    """

    @staticmethod
    def status(path):
        return dict(line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line)

    @staticmethod
    def start_time(path):
        stat = (path / "stat").read_text()
        return int(stat[stat.rfind(")") + 2:].split()[19])

    def __init__(self, pid, command, application):
        self.pid, self.command = pid, command
        self.application = Path(application).resolve()
        self.path = None
        self.peak = None
        self.samples = 0
        self.attempts = 0
        self.evidence = {"popen_pid": pid, "scope": "Validated application process; not process-tree sum",
                         "available": False, "reason": "Linux /proc unavailable"}
        self.enabled = False
        if platform.system() != "Linux":
            return
        try:
            own = self.status(Path("/proc/self"))
            self.namespace = os.readlink("/proc/self/ns/pid")
            self.parent_pid = int(own["Pid"])
            self.parent_nspid = list(map(int, own["NSpid"].split()))
            if self.parent_nspid[-1] != os.getpid():
                raise RuntimeError("Cannot correlate harness PID to mounted /proc namespace")
            app_stat = self.application.stat()
            self.inode, self.device = app_stat.st_ino, app_stat.st_dev
            self.evidence["harness"] = {"os_pid": os.getpid(), "proc_pid": self.parent_pid,
                                        "nspid": self.parent_nspid, "pid_namespace": self.namespace}
            self.evidence["reason"] = "Application identity has not yet been verified"
            self.enabled = True
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            self.evidence["reason"] = str(error)

    def resolve(self):
        direct = Path("/proc") / str(self.pid)
        for path in (direct, *[p for p in Path("/proc").iterdir() if p.name.isdigit() and p != direct]):
            try:
                status = self.status(path)
                nspid = list(map(int, status.get("NSpid", "").split()))
                if (not nspid or nspid[-1] != self.pid or int(status["PPid"]) != self.parent_pid
                        or os.readlink(path / "ns/pid") != self.namespace):
                    continue
                self.path = path
                self.started = self.start_time(path)
                return
            except (OSError, KeyError, ValueError, IndexError):
                continue
        self.evidence["reason"] = "Cannot locate child with matching NSpid, namespace and parent"

    def sample(self):
        if not self.enabled:
            return None
        try:
            if self.path is None:
                # Resolution should be immediate; do not scan /proc forever if
                # the host hides identity metadata or the child has disappeared.
                self.attempts += 1
                if self.attempts > 20:
                    return None
                self.resolve()
                if self.path is None:
                    return None
            status = self.status(self.path)
            if (self.start_time(self.path) != self.started or int(status["PPid"]) != self.parent_pid
                    or list(map(int, status["NSpid"].split()))[-1] != self.pid
                    or os.readlink(self.path / "ns/pid") != self.namespace):
                raise RuntimeError("Resolved process identity changed")
            raw_command = (self.path / "cmdline").read_bytes()
            # Remove one terminator only: an explicitly empty final argument
            # (the default negative prompt) is a second trailing NUL.
            command = [os.fsdecode(part) for part in raw_command[:-1].split(b"\0")] if raw_command.endswith(b"\0") else []
            executable = os.readlink(self.path / "exe")
            comm = (self.path / "comm").read_text().rstrip("\n")
            if (command != self.command or Path(executable).resolve() != Path(self.command[0]).resolve()
                    or comm != self.application.name[:15]):
                raise RuntimeError("Child has not established the expected application command/executable/comm")
            mapping = None
            for line in (self.path / "maps").read_text().splitlines():
                fields = line.split(None, 5)
                major, minor = (int(value, 16) for value in fields[3].split(":"))
                if ("x" in fields[1] and int(fields[4]) == self.inode
                        and (major, minor) == (os.major(self.device), os.minor(self.device))):
                    mapping = line
                    break
            if mapping is None:
                raise RuntimeError("No executable mapping of the expected application inode")
            match = re.fullmatch(r"\s*(\d+) kB", status.get("VmHWM", ""))
            if not match:
                raise RuntimeError("Validated application has no VmHWM record")
            value = int(match.group(1)) * 1024
            self.peak = max(self.peak or 0, value)
            self.samples += 1
            self.evidence.update({"available": True, "reason": None, "validated_samples": self.samples,
                                  "process": {"proc_pid": int(self.path.name), "nspid": list(map(int, status["NSpid"].split())),
                                              "parent_proc_pid": self.parent_pid, "pid_namespace": self.namespace,
                                              "start_time_ticks": self.started, "comm": comm,
                                              "executable": executable, "command": command,
                                              "application_executable_mapping": mapping}})
            return value
        except (OSError, KeyError, ValueError, IndexError, RuntimeError) as error:
            self.evidence["last_unavailable_sample"] = str(error)
            if not self.samples:
                self.evidence["reason"] = str(error)
            return None


def run_child(command, directory, timeout, software_threads, report):
    started = time.monotonic()
    stdout, stderr = directory / "inference.stdout.log", directory / "inference.stderr.log"
    report.update({"command": command, "stdout_log": stdout.name, "stderr_log": stderr.name,
                   "expected_application_exit_code": 0, "expected_host_exit_status": 0})
    with tempfile.TemporaryDirectory(prefix="runtime-", dir=directory) as temporary:
        environment = runtime_environment(Path(temporary))
        environment["LP_NUM_THREADS"] = str(software_threads)
        # Separate regular files preserve both streams on Cosmopolitan Windows;
        # aliasing stdout/stderr file handles can overwrite their independent offsets.
        with stdout.open("wb") as output, stderr.open("wb") as errors:
            process = subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=errors,
                                       env=environment)
            memory = ProcessMemory(process.pid, command, command[1] if os.name != "nt" else command[0])
            try:
                while True:
                    memory.sample()
                    code = process.poll()
                    if code is not None:
                        break
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(f"Inference exceeded {timeout} seconds")
                    time.sleep(0.1)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
                report["exit_code"] = process.returncode
                report["elapsed_seconds"] = round(time.monotonic() - started, 3)
                report["peak_rss_bytes"] = memory.peak
                report["memory_sampling"] = memory.evidence
                report["peak_rss_measurement"] = ("sampled Linux /proc VmHWM at 100ms; may miss final peak"
                                                   if memory.peak is not None else "unavailable; application process identity could not be verified")
    if code != 0:
        raise RuntimeError(f"Inference failed with native exit status {code}; see separate logs")
    return stdout.read_text(errors="replace"), stderr.read_text(errors="replace")


def verify(args):
    directory = args.artifact_dir.resolve()
    model = args.model.resolve()
    logs = args.output_dir.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    image = logs / "image.png"
    report = {"status": "failed", "platform": platform.platform(), "backend": args.backend,
              "scope": "Real trained SD1.5 inference and decoded PNG plumbing; visual quality not assessed",
              "low_step_plumbing": args.steps <= 4, "filesystem_isolated": False,
              "software_driver_workers": {"environment": "LP_NUM_THREADS", "value": args.software_threads},
              "parameters": {"prompt": args.prompt, "negative_prompt": args.negative_prompt,
                             "width": args.width, "height": args.height, "steps": args.steps,
                             "seed": args.seed, "threads": args.threads, "cfg_scale": args.cfg_scale},
              "output_png": image.name}
    app = directory / APPLICATION
    try:
        pin, report["model_pin"] = read_model_pin(args)
        report["model"] = {"repository": pin["repository"], "revision": pin["revision"],
                           "filename": pin["filename"], "bytes": pin["bytes"],
                           "expected_sha256": pin["sha256"], "path": str(model)}
        if image.exists():
            raise RuntimeError("Output image already exists; use a fresh --output-dir")
        if not model.is_file() or model.stat().st_size != pin["bytes"]:
            raise RuntimeError("Model size differs from the pinned full SD1.5 checkpoint")
        report["model"]["sha256_before"] = sha256(model)
        if report["model"]["sha256_before"] != pin["sha256"]:
            raise RuntimeError("Model SHA256 differs from the pinned trained SD1.5 checkpoint")
        expected = args.expected_sha256 or os.environ.get("EXPECTED_SHA256")
        report["sha256_before"] = verify_package(directory, expected)
        command = [str(app)] if os.name == "nt" else [str(directory / LOADER), str(app)]
        command += ["image", "--model", str(model), "--prompt", args.prompt,
                    "--output", str(image), "--backend", args.backend,
                    "--width", str(args.width), "--height", str(args.height),
                    "--steps", str(args.steps), "--seed", str(args.seed),
                    "--threads", str(args.threads), "--cfg-scale", str(args.cfg_scale),
                    "--negative-prompt", args.negative_prompt]
        for flag, value in (("--sampler", args.sampler), ("--scheduler", args.scheduler)):
            if value is not None:
                command += [flag, value]
        report["launch_mode"] = "native Windows PE" if os.name == "nt" else "explicit APE ELF loader"
        try:
            text, stderr_text = run_child(command, logs, args.timeout, args.software_threads, report)
        finally:
            report["sha256_after"] = sha256(app)
            report["model"]["sha256_after"] = sha256(model)
            report["app_hash_unchanged"] = report["sha256_after"] == report["sha256_before"]
            report["model_hash_unchanged"] = report["model"]["sha256_after"] == pin["sha256"]
        if not report["app_hash_unchanged"] or not report["model_hash_unchanged"]:
            raise RuntimeError("Application or model bytes changed during inference")
        report["png"] = inspect_png(image, args.width, args.height)
        report["execution"] = verify_markers(text, stderr_text, args, report["png"])
        report["status"] = "passed"
        return 0
    except Exception as error:
        report["error"] = str(error)
        print(f"Inference verification failed: {error}", file=sys.stderr)
        return 1
    finally:
        report["evidence"] = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                              for path in (logs / "inference.stdout.log", logs / "inference.stderr.log", image)
                              if path.is_file()]
        (logs / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-pin", type=Path,
                        help="Model metadata; default: artifact/verifier PIN.json, then repository inference/PIN.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "webgpu"), required=True)
    parser.add_argument("--expected-sha256", help="Expected application SHA256 (the model SHA256 comes from PIN.json)")
    parser.add_argument("--prompt", default="a photograph of a red apple on a wooden table, natural light")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--software-threads", type=int, default=2,
                        help="LP_NUM_THREADS software Vulkan worker limit, separate from GGML --threads")
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--sampler")
    parser.add_argument("--scheduler")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if any(value < 64 or value > 1024 or value % 64 for value in (args.width, args.height)):
        parser.error("Verification width/height must be multiples of 64 in 64..1024")
    if not 2 <= args.steps <= 100 or not 1 <= args.threads <= 64 or not 0 <= args.seed < 2**63:
        parser.error("Verification bounds: steps 2..100 (sampling interval required), threads 1..64, seed 0..INT64_MAX")
    if not 1 <= args.software_threads <= 256:
        parser.error("Software driver threads must be in 1..256")
    if not math.isfinite(args.cfg_scale) or not 0 <= args.cfg_scale <= 100:
        parser.error("CFG scale must be finite and in 0..100")
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 14400:
        parser.error("Timeout must be finite and in 1..14400 seconds")
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
