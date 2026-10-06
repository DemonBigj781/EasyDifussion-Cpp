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
    "DIFFUSION_SELFTEST PASS", "COSMOPOLITAN_SELFTEST PASS",
)


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
    if result.returncode != expected_exit:
        print(text[-18000:])
        raise RuntimeError(f"{label} failed with exit code {result.returncode}; see {path}")
    return text, {"name": label, "arguments": arguments, "exit_code": result.returncode,
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
        process = subprocess.Popen(command_for(root, ["sdkit", "--backend", "cpu", "--port", str(port)], isolate),
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
            report["tests"].append(result)
            for label, arguments in [
                ("devices", ["sdkit", "--list-devices"]),
                ("diffusion-command", ["sdkit", "--help"]),
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
