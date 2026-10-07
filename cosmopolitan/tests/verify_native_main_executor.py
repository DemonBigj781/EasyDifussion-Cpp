#!/usr/bin/env python3
"""Build/run the Linux host-libc TLS lane regression; no model or GPU required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PORT = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=PORT / "out/sdk")
    parser.add_argument("--output", type=Path, default=PORT / "out/native-main-executor-test")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("This regression requires x86-64 Linux; no skipped pass")
    sdk, output = args.sdk.resolve(), args.output.resolve()
    cc = shutil.which("cc")
    if not cc:
        raise RuntimeError("A host C compiler is required by the fixture and cosmo_dlopen")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "tmp"
    temporary.mkdir(exist_ok=True)
    fixture = output / "libnative_tls_fixture.so"
    executable = output / "native-main-executor-test.com"
    # Use only the logging declarations from prepared headers; all SD code is
    # replaced by three test TLS accessors in the fixture program.
    includes = [PORT / "src", PORT / "backend/include",
                PORT / "out/source/sdkit3-port-source/include",
                PORT.parent / "source/sdkit3-port-source/stable-diffusion.cpp/include"]
    if not (includes[2] / "logging.h").is_file():
        raise RuntimeError("Run build.py --prepare-only first (patched logging.h required)")
    commands = [
        [cc, "-shared", "-fPIC", "-O2", "-pthread", str(HERE / "native_tls_fixture.c"), "-o", str(fixture)],
        [str(sdk / "bin/x86_64-unknown-cosmo-c++"), "-std=c++20", "-O2", "-g", "-pthread",
         *["-I" + str(path) for path in includes], str(PORT / "src/native_main_executor.cpp"),
         str(PORT / "src/inference_worker.cpp"), str(HERE / "native_main_executor_test.cpp"), "-o", str(executable)],
    ]
    for command in commands:
        subprocess.run(command, check=True, timeout=120)
    before = sha(executable)
    command = [str(sdk / "bin/ape-x86_64.elf"), str(executable), str(fixture)]
    env = {"PATH": os.pathsep.join(dict.fromkeys([str(Path(cc).parent), "/usr/bin", "/bin"])),
           "TMPDIR": str(temporary)}
    result = subprocess.run(command, capture_output=True, timeout=30, env=env)
    (output / "stdout.log").write_bytes(result.stdout)
    (output / "stderr.log").write_bytes(result.stderr)
    after = sha(executable)
    text = result.stdout.decode(errors="replace")
    stderr = result.stderr.decode(errors="replace")
    success = (result.returncode == 0 and before == after
               and "NATIVE_MAIN_EXECUTOR checks=14 PASS\n" in text
               and "NATIVE_LANE_VULKAN extensions=" in text
               and stderr.count("NATIVE_INFERENCE_MAIN stack_bytes=") == 2
               and stderr.count("NATIVE_HTTP_SERVICE stack_bytes=") == 2
               and "NATIVE_INFERENCE_WORKER" not in stderr)
    report = {"success": success, "scope": "host TLS, native Vulkan loader enumeration, synchronous main lane, exception/OOM transfer and shutdown; no shader/GPU proof",
              "command": command, "returncode": result.returncode,
              "sha256_before": before, "sha256_after": after,
              "fixture_sha256": sha(fixture), "build_commands": commands,
              "stdout_sha256": sha(output / "stdout.log"), "stderr_sha256": sha(output / "stderr.log")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("native main executor regression: " + str(error), file=sys.stderr)
        sys.exit(1)
