#!/usr/bin/env python3
"""Run the packaged native application from a fresh directory on either OS.

Python is a host-side CI harness. On Linux --isolate replaces the child with
the APE loader after a real chroot; no Python or shared libraries are copied
into the application's root. This checks deployment dependencies, not a
security boundary for untrusted programs.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


APPLICATION = "easy-diffusion.exe"
LOADER = "ape-x86_64.elf"
MODEL_SHA256 = "047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b"
# Independently captured with the original llama/GGML, before merging SD's tree.
EXPECTED_GREEDY_IDS = [432, 383, 286, 261, 376, 298, 315, 421,
                       395, 317, 426, 338, 401, 396, 267, 337]
SELFTEST_MARKERS = (
    "GGML_SELFTEST PASS", "SHARED_GGML_SELFTEST PASS", "LLAMA_SELFTEST PASS", "TRAINING_MATH_SELFTEST PASS",
    "DIFFUSION_SELFTEST PASS", "WEBGPU_GGML_SELFTEST PASS", "WEBGPU_LLAMA_SELFTEST PASS",
    "WEBGPU_INPLACE_SELFTEST PASS", "COSMOPOLITAN_SELFTEST PASS",
)


def validate_webgpu_output(text):
    """Require scoped GGML dispatch/readback and actual trained-model inference."""
    lines = text.splitlines()
    expected_cases = {(kind, iteration) for kind in ("F32", "Q4_0") for iteration in (0, 1)}
    counter_pattern = (r"graphs=(\d+) submissions=(\d+) dispatches=(\d+) "
                       r"matmuls=(\d+) readbacks=(\d+) native_loader_opens=0")

    def counters(match, start, minimum_readbacks):
        names = ("graphs", "submissions", "dispatches", "matmuls", "readbacks")
        result = dict(zip(names, map(int, match.groups()[start:])))
        if any(result[name] <= 0 for name in names) or result["readbacks"] < minimum_readbacks:
            raise RuntimeError("WebGPU execution lacks graph, dispatch, submission or readback evidence")
        if result["matmuls"] > result["dispatches"]:
            raise RuntimeError("WebGPU matmul count exceeds actual dispatch count")
        return result

    cases = {}
    for line in (line for line in lines if line.startswith("WEBGPU_GGML_EXECUTION")):
        match = re.fullmatch(r"WEBGPU_GGML_EXECUTION type=(F32|Q4_0) iteration=([01]) "
                             r"provider=embedded software=1 " + counter_pattern + r" cpu_fallback=0", line)
        if not match:
            raise RuntimeError("Invalid or non-embedded WebGPU GGML execution result")
        key = (match.group(1), int(match.group(2)))
        if key in cases:
            raise RuntimeError("Duplicate WebGPU graph execution result")
        cases[key] = {"type": key[0], "iteration": key[1], **counters(match, 2, 3)}
    if set(cases) != expected_cases:
        raise RuntimeError("Missing actual WebGPU F32/Q4 graph execution cases")

    checks = set()
    for line in (line for line in lines if line.startswith("WEBGPU_CHECK")):
        match = re.fullmatch(r"WEBGPU_CHECK type=(F32|Q4_0) iteration=([01]) "
                             r"stage=(matmul|bias-rmsnorm-silu|softmax) values=143 "
                             r"max_error=([0-9.eE+-]+) PASS", line)
        if not match:
            raise RuntimeError("Invalid WebGPU numerical readback result")
        error = float(match.group(4))
        if not math.isfinite(error) or error < 0:
            raise RuntimeError("Non-finite WebGPU numerical error")
        key = (match.group(1), int(match.group(2)), match.group(3))
        if key in checks:
            raise RuntimeError("Duplicate WebGPU numerical readback result")
        checks.add(key)
    if checks != {(kind, iteration, stage) for kind, iteration in expected_cases
                  for stage in ("matmul", "bias-rmsnorm-silu", "softmax")}:
        raise RuntimeError("Missing WebGPU scalar-reference readback checks")
    unsupported = [line for line in lines if line.startswith("WEBGPU_UNSUPPORTED")]
    if unsupported != ["WEBGPU_UNSUPPORTED op=SILU_BACK supported=0"]:
        raise RuntimeError("WebGPU unsupported-operator rejection check is missing")

    id_lines = [line for line in lines if line.startswith("WEBGPU_LLAMA_GREEDY_IDS")]
    match = re.fullmatch(r"WEBGPU_LLAMA_GREEDY_IDS ([0-9]+(?:,[0-9]+)*)", id_lines[0]) if len(id_lines) == 1 else None
    if not match or [int(token) for token in match.group(1).split(",")] != EXPECTED_GREEDY_IDS:
        raise RuntimeError("WebGPU llama differs from the independent 16-token reference")
    detail = "WEBGPU_LLAMA_SELFTEST prompt_tokens=5 decode_steps=16 vocab=512 finite_logits=1 threads=1"
    if lines.count(detail) != 1:
        raise RuntimeError("WebGPU llama did not complete the required finite-logit decode steps")
    execution = [line for line in lines if line.startswith("WEBGPU_LLAMA_EXECUTION")]
    match = re.fullmatch(r"WEBGPU_LLAMA_EXECUTION provider=embedded software=1 " + counter_pattern,
                         execution[0]) if len(execution) == 1 else None
    if not match:
        raise RuntimeError("Missing actual embedded WebGPU llama execution")
    inference = counters(match, 0, 1)
    decode = [line for line in lines if line.startswith("WEBGPU_LLAMA_DECODE_EXECUTION")]
    match = re.fullmatch(r"WEBGPU_LLAMA_DECODE_EXECUTION steps=16 graphs=(\d+) "
                         r"submissions=(\d+) dispatches=(\d+) matmuls=(\d+) readbacks=(\d+)",
                         decode[0]) if len(decode) == 1 else None
    if not match:
        raise RuntimeError("Missing WebGPU llama execution scoped to the sixteen decode steps")
    decode_execution = counters(match, 0, 1)
    if any(decode_execution[name] > inference[name] for name in decode_execution):
        raise RuntimeError("WebGPU decode count exceeds the full inference count")
    for marker in ("WEBGPU_GGML_SELFTEST PASS", "WEBGPU_LLAMA_SELFTEST PASS"):
        if lines.count(marker) != 1:
            raise RuntimeError(f"Expected one WebGPU success result: {marker}")
    return {"provider": "embedded", "software_adapter": True, "native_loader_opens": 0,
            "graph_cpu_fallback": False, "graph_cases": [cases[key] for key in sorted(cases)],
            "scalar_readback_checks": len(checks), "llama_execution": inference,
            "llama_decode_execution": {"steps": 16, **decode_execution},
            "llama_greedy_token_ids": EXPECTED_GREEDY_IDS, "independent_reference_match": True}


def validate_inplace_output(text):
    """Require real mixed-backend and preallocated-device alias regressions."""
    lines = text.splitlines()
    cases = {}
    for line in (line for line in lines if line.startswith("WEBGPU_INPLACE_EXECUTION")):
        match = re.fullmatch(
            r"WEBGPU_INPLACE_EXECUTION owner=(scheduler|preallocated|preallocated-webgpu) iteration=([01]) "
            r"alias_nodes=(\d+) alias_backend=(CPU|WebGPU) consumer_backend=WebGPU consumer_host_buffer=0 "
            r"cpu_fallback=([01]) graphs=(\d+) submissions=(\d+) dispatches=(\d+) matmuls=(\d+) "
            r"readbacks=(\d+) native_loader_opens=0 PASS", line)
        if not match:
            raise RuntimeError("Invalid in-place storage execution record")
        owner, iteration = match.group(1), int(match.group(2))
        gpu_owner = owner == "preallocated-webgpu"
        if (int(match.group(3)), match.group(4), int(match.group(5))) != (
                2 if gpu_owner else 5, "WebGPU" if gpu_owner else "CPU", 0 if gpu_owner else 1):
            raise RuntimeError("In-place owner placement/fallback differs from the required case")
        counters = dict(zip(("graphs", "submissions", "dispatches", "matmuls", "readbacks"),
                            map(int, match.groups()[5:])))
        if any(value <= 0 for value in counters.values()) or counters["matmuls"] > counters["dispatches"]:
            raise RuntimeError("In-place regression lacks actual WebGPU execution/readback")
        key = (owner, iteration)
        if key in cases:
            raise RuntimeError("Duplicate in-place storage case")
        cases[key] = {"owner": owner, "iteration": iteration, "cpu_fallback": not gpu_owner, **counters}
    expected = {(owner, iteration) for owner in ("scheduler", "preallocated", "preallocated-webgpu")
                for iteration in (0, 1)}
    if set(cases) != expected:
        raise RuntimeError("Missing scheduler-allocated or preallocated in-place storage cases")
    stages = {"cpu-alias-activation": 128, "webgpu-projection-f16": 24,
              "webgpu-owner-silu": 128, "webgpu-owner-projection-f16": 24}
    checks = set()
    for line in (line for line in lines if line.startswith("WEBGPU_INPLACE_CHECK")):
        match = re.fullmatch(r"WEBGPU_INPLACE_CHECK owner=(scheduler|preallocated) iteration=([01]) "
                             r"stage=(\S+) values=(\d+) max_error=([0-9.eE+-]+) PASS", line)
        if not match or stages.get(match.group(3)) != int(match.group(4)):
            raise RuntimeError("Invalid in-place numerical readback record")
        error = float(match.group(5))
        if not math.isfinite(error) or error < 0:
            raise RuntimeError("Non-finite in-place numerical error")
        key = (match.group(1), int(match.group(2)), match.group(3))
        if key in checks:
            raise RuntimeError("Duplicate in-place numerical check")
        checks.add(key)
    expected_checks = {(owner, iteration, stage) for owner in ("scheduler", "preallocated")
                       for iteration in (0, 1) for stage in ("cpu-alias-activation", "webgpu-projection-f16")}
    expected_checks |= {("preallocated", iteration, stage) for iteration in (0, 1)
                        for stage in ("webgpu-owner-silu", "webgpu-owner-projection-f16")}
    if checks != expected_checks or lines.count("WEBGPU_INPLACE_SELFTEST PASS") != 1:
        raise RuntimeError("Incomplete in-place scalar-reference regression")
    return {"cases": [cases[key] for key in sorted(cases)], "scalar_readback_checks": len(checks)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_package(directory, expected):
    executable = directory / APPLICATION
    actual = sha256(executable)
    if expected and actual != expected.lower():
        raise RuntimeError("Application differs from the build job's SHA-256")
    manifest = json.loads((directory / "BUILD.json").read_text())
    if actual != manifest["executable"]["sha256"]:
        raise RuntimeError("Application differs from BUILD.json")
    data = executable.read_bytes()
    if data[:2] != b"MZ":
        raise RuntimeError("Missing PE/APE header")
    pe, = struct.unpack_from("<I", data, 0x3C)
    if data[pe:pe + 4] != b"PE\0\0" or struct.unpack_from("<H", data, pe + 4)[0] != 0x8664:
        raise RuntimeError("Application is not an x86-64 Windows PE")
    with zipfile.ZipFile(executable) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise RuntimeError("Duplicate embedded resource names")
        for name, info in manifest["embedded_resources"].items():
            contents = archive.read(name)
            if len(contents) != info["bytes"] or hashlib.sha256(contents).hexdigest() != info["sha256"]:
                raise RuntimeError(f"Embedded resource mismatch: {name}")
        if hashlib.sha256(archive.read("models/stories260K.gguf")).hexdigest() != MODEL_SHA256:
            raise RuntimeError("Embedded model is not the pinned trained fixture")
    return actual


def runtime_environment(work):
    environment = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "TERM": "dumb"}
    if os.name == "nt":
        system = os.environ["SystemRoot"]
        environment.update({"SystemRoot": system, "WINDIR": system,
                            "PATH": str(Path(system) / "System32"),
                            "TEMP": str(work), "TMP": str(work)})
    else:
        environment.update({"PATH": "/no-tools", "TMPDIR": str(work)})
    return environment


def isolation_worker(arguments):
    if len(arguments) < 2 or platform.system() != "Linux":
        raise RuntimeError("Invalid isolation worker invocation")
    root, command = Path(arguments[0]), arguments[1:]
    if not root.is_absolute():
        raise RuntimeError("Isolation root must be absolute")
    os.chroot(root)
    os.chdir("/tmp")
    if set(os.listdir("/")) != {APPLICATION, LOADER, "tmp"}:
        raise RuntimeError("Unexpected file in isolated application root")
    os.execve("/" + LOADER, ["/" + LOADER, "/" + APPLICATION, *command],
              runtime_environment(Path("/tmp")))


def command_for(root, arguments, isolate):
    if isolate:
        return [sys.executable, "-I", "-S", str(Path(__file__).resolve()),
                "--_isolation-worker", str(root), *arguments]
    if os.name == "nt":
        return [str(root / APPLICATION), *arguments]
    return [str(root / LOADER), str(root / APPLICATION), *arguments]


@contextmanager
def capture_logs(path):
    # On Windows, aliased regular-file stdout/stderr handles can acquire
    # independent offsets in the child and overwrite each other's output.
    # Capture separate files and assemble diagnostics only in the parent.
    # The combined log groups streams; it does not claim chronological order.
    stdout_path = path.with_suffix(".stdout.log")
    stderr_path = path.with_suffix(".stderr.log")
    try:
        with stdout_path.open("wb") as output, stderr_path.open("wb") as errors:
            yield output, errors
    finally:
        with path.open("wb") as combined:
            for label, source in ((b"stdout", stdout_path), (b"stderr", stderr_path)):
                combined.write(b"\n--- " + label + b" ---\n")
                if source.is_file():
                    with source.open("rb") as stream:
                        shutil.copyfileobj(stream, combined)


def run_command(root, arguments, isolate, logs, label, timeout=240, expected_exit=0):
    started = time.monotonic()
    path = logs / (label + ".log")
    with capture_logs(path) as (output, errors):
        result = subprocess.run(command_for(root, arguments, isolate),
                                cwd=root / "tmp", env=runtime_environment(root / "tmp"),
                                stdin=subprocess.DEVNULL, stdout=output,
                                stderr=errors, timeout=timeout)
    text = path.read_text(errors="replace")
    # The pinned SDK's Windows _Exit passes (exitcode << 8) to process
    # termination. Compare exact native status; do not normalize failures.
    # https://github.com/jart/cosmopolitan/blob/4.0.2/libc/intrin/exit.c#L87-L105
    expected_host_status = expected_exit << 8 if os.name == "nt" else expected_exit
    if result.returncode != expected_host_status:
        print(text[-18000:])
        raise RuntimeError(f"{label} failed with native exit status {result.returncode}; "
                           f"expected {expected_host_status} for application exit code {expected_exit}; see {path}")
    return text, {"name": label, "arguments": arguments, "exit_code": result.returncode,
                  "expected_application_exit_code": expected_exit,
                  "expected_host_exit_status": expected_host_status,
                  "elapsed_seconds": round(time.monotonic() - started, 3), "log": path.name,
                  "stdout_log": path.with_suffix(".stdout.log").name,
                  "stderr_log": path.with_suffix(".stderr.log").name}


def http_checks(root, isolate, logs):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    # Never route a loopback verification request through a host proxy.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def fetch(path):
        with client.open(base + path, timeout=5) as response:
            return response.status, response.headers, response.read()

    path = logs / "server.log"
    with capture_logs(path) as (output, errors):
        process = subprocess.Popen(command_for(root, ["sdkit", "--backend", "cpu", "--provider", "embedded", "--port", str(port)], isolate),
                                   cwd=root / "tmp", env=runtime_environment(root / "tmp"),
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=errors)
        try:
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited before startup; see {path}")
                try:
                    status, _, ping = fetch("/v1/internal/ping")
                    if status == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Server did not start; see {path}")
                time.sleep(0.1)
            checks = []
            for route, expected_type, required in [
                ("/", "text/html", b"/cpp-ui/assets/ui.css"),
                ("/cpp-ui/training", "text/html", b"page-content"),
                ("/cpp-ui/assets/ui.css", "text/css", b"{"),
                ("/cpp-ui/scripts/generate.js", "text/javascript", b"/v1/sdapi/v1/txt2img"),
                ("/cpp-ui/scripts/kiosk.js", "text/javascript", b"cosmopolitan-capabilities"),
                ("/v1/sdapi/v1/cosmopolitan-capabilities", "application/json", b"native-single-user"),
                ("/v1/sdapi/v1/checkpoints", "application/json", b"models"),
                ("/v1/sdapi/v1/backend-devices", "application/json", b"cpu"),
            ]:
                status, headers, body = fetch(route)
                if status != 200 or expected_type not in headers.get("Content-Type", ""):
                    raise RuntimeError(f"Unexpected HTTP status/type for {route}")
                if required not in body.lower():
                    raise RuntimeError(f"Missing expected HTTP content for {route}")
                checks.append({"path": route, "status": status, "bytes": len(body)})
            try:
                status, _, _ = fetch("/cpp-ui/assets/%2e%2e/LICENSE")
            except urllib.error.HTTPError as error:
                status = error.code
            if status not in (400, 404):
                raise RuntimeError("Embedded asset route accepted parent traversal")
            return {"name": "native-server-and-embedded-ui", "ping_status": 200,
                    "checks": checks, "parent_traversal_status": status,
                    "log": path.name,
                    "stdout_log": path.with_suffix(".stdout.log").name,
                    "stderr_log": path.with_suffix(".stderr.log").name}
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def verify(args):
    directory = args.artifact_dir.resolve()
    if args.isolate and platform.system() != "Linux":
        raise RuntimeError("--isolate requires Linux with real chroot capability")
    logs = directory / ("results-linux-isolated" if args.isolate else "results-" + platform.system().lower())
    logs.mkdir(exist_ok=True)
    report = {"status": "failed", "platform": platform.platform(), "isolated": args.isolate,
              "tests": [], "scope": "Native integration checks; not full application, image or training parity"}
    try:
        expected = args.expected_sha256 or os.environ.get("EXPECTED_SHA256")
        original = verify_package(directory, expected)
        report["sha256_before"] = original
        with tempfile.TemporaryDirectory(prefix="easy-diffusion-runtime-") as temporary:
            root = Path(temporary)
            for name in ((APPLICATION, LOADER) if os.name != "nt" else (APPLICATION,)):
                shutil.copyfile(directory / name, root / name)
                (root / name).chmod(0o755)
            (root / "tmp").mkdir()
            report["initial_root_entries"] = sorted(p.name for p in root.iterdir())
            text, result = run_command(root, ["--self-test"], args.isolate, logs, "self-test")
            for marker in SELFTEST_MARKERS:
                if marker not in text.splitlines():
                    raise RuntimeError(f"Missing actual self-test result: {marker}")
            if not re.search(r"^UI_SELFTEST pages=16 embedded_assets=yes PASS$", text, re.M):
                raise RuntimeError("Embedded C++ UI rendering self-test is missing")
            id_lines = [line for line in text.splitlines()
                        if line == "LLAMA_GREEDY_IDS" or line.startswith("LLAMA_GREEDY_IDS ")]
            match = re.fullmatch(r"LLAMA_GREEDY_IDS ([0-9]+(?:,[0-9]+)*)", id_lines[0]) if len(id_lines) == 1 else None
            if not match or [int(token) for token in match.group(1).split(",")] != EXPECTED_GREEDY_IDS:
                raise RuntimeError("Trained llama fixture differs from the independent 16-token reference")
            result["greedy_token_ids"] = EXPECTED_GREEDY_IDS
            result["independent_reference_match"] = True
            result["webgpu"] = validate_webgpu_output(text)
            result["inplace_storage"] = validate_inplace_output(text)
            report["tests"].append(result)
            for label, arguments in [
                ("devices", ["sdkit", "--list-devices", "--provider", "embedded"]),
                ("diffusion-command", ["sdkit", "--help"]),
                ("image-command", ["image", "--help"]),
                ("training-command", ["train", "--help"]),
            ]:
                text, result = run_command(root, arguments, args.isolate, logs, label)
                if not text.strip():
                    raise RuntimeError(f"{label} returned no output")
                report["tests"].append(result)
            text, result = run_command(root, ["train", "--model", "unused", "--dataset", "unused",
                                               "--output", "unused", "--device", "CPU", "--threads"],
                                       args.isolate, logs, "training-unmatched-option", expected_exit=2)
            if "Missing value for native trainer option" not in text:
                raise RuntimeError("Trainer did not reject an unmatched option before starting work")
            report["tests"].append(result)
            for label, arguments, required in [
                ("image-unmatched-option", ["image", "--model"], "unknown, duplicate, or incomplete option"),
                ("image-invalid-dimensions", ["image", "--width", "65"], "invalid value for --width"),
                ("image-invalid-backend", ["image", "--backend", "missing"], "compute.backend must be cpu or webgpu"),
            ]:
                text, result = run_command(root, arguments, args.isolate, logs, label, expected_exit=2)
                if required not in text:
                    raise RuntimeError(f"{label} did not reject the invalid request before model loading")
                report["tests"].append(result)
            report["tests"].append(http_checks(root, args.isolate, logs))
            if sha256(root / APPLICATION) != original:
                raise RuntimeError("Application changed while running")
        report["sha256_after"] = sha256(directory / APPLICATION)
        if report["sha256_after"] != original:
            raise RuntimeError("Source artifact changed while running")
        report["status"] = "passed"
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        report["error"] = str(error)
        print(f"Runtime verification failed: {error}", file=sys.stderr)
        return 1
    finally:
        (logs / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_isolation-worker":
        isolation_worker(sys.argv[2:])
        return 125
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--isolate", action="store_true")
    return verify(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
