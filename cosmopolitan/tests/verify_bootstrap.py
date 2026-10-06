#!/usr/bin/env python3
"""Verify Linux shell bootstrap from the single packaged application file.

The application is launched by /bin/sh with a fresh TMPDIR and a PATH containing
only the host utilities needed by the APE shell header. Its embedded ELF loader
is extracted transiently into that directory. This check uses host shell tools;
the separate verify_runtime.py --isolate test checks explicit-loader execution
inside an empty filesystem. No external APE loader is copied for this test.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time

from verify_runtime import (
    APPLICATION,
    EXPECTED_GREEDY_IDS,
    SELFTEST_MARKERS,
    sha256,
    verify_package,
)


UTILITIES = ("uname", "mkdir", "dd", "gzip", "chmod", "mv")


def validate_selftest_output(text):
    lines = text.splitlines()
    for marker in (*SELFTEST_MARKERS, "UI_SELFTEST pages=16 embedded_assets=yes PASS"):
        if lines.count(marker) != 1:
            raise RuntimeError(f"Expected one actual self-test result: {marker}")
    # Count all marker lines before validating their syntax, so a second,
    # malformed marker cannot hide behind a successfully parsed first line.
    id_lines = [line for line in lines if line.startswith("LLAMA_GREEDY_IDS")]
    match = re.fullmatch(r"LLAMA_GREEDY_IDS ([0-9]+(?:,[0-9]+)*)", id_lines[0]) if len(id_lines) == 1 else None
    if not match:
        raise RuntimeError("Expected exactly one well-formed LLAMA_GREEDY_IDS result")
    token_ids = [int(token) for token in match.group(1).split(",")]
    if token_ids != EXPECTED_GREEDY_IDS:
        raise RuntimeError("Bootstrap inference differs from the independent 16-token reference")
    return token_ids


def extracted_loader(runtime):
    helpers = sorted(runtime.glob(".ape-*"))
    if len(helpers) != 1:
        raise RuntimeError(f"Expected one extracted .ape-* loader, found {len(helpers)}")
    helper = helpers[0]
    info = helper.lstat()
    if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
        raise RuntimeError("Extracted APE loader is not an executable regular file")
    with helper.open("rb") as stream:
        header = stream.read(64)
    if (len(header) != 64 or header[:7] != b"\x7fELF\x02\x01\x01"
            or struct.unpack_from("<H", header, 18)[0] != 62):
        raise RuntimeError("Extracted APE loader is not an x86-64 ELF")
    return {"filename": helper.name, "bytes": info.st_size, "sha256": sha256(helper),
            "format": "x86-64 ELF", "transient_extraction": True}


def verify(args):
    directory = args.artifact_dir.resolve()
    logs = directory / "results-linux-bootstrap"
    logs.mkdir(exist_ok=True)
    report = {
        "status": "failed",
        "platform": platform.platform(),
        "scope": "Linux /bin/sh bootstrap with host utilities and transient embedded-loader extraction",
        "filesystem_isolated": False,
        "explicit_external_loader_supplied": False,
        "path_utilities": list(UTILITIES),
        "log": "self-test.log",
    }
    log = logs / report["log"]
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
            raise RuntimeError("This bootstrap check requires x86-64 Linux")
        expected = args.expected_sha256 or os.environ.get("EXPECTED_SHA256")
        original = verify_package(directory, expected)
        report["sha256_before"] = original
        source = directory / APPLICATION
        with tempfile.TemporaryDirectory(prefix="easy-diffusion-bootstrap-") as temporary:
            root = Path(temporary)
            commands, runtime = root / "tools", root / "runtime"
            commands.mkdir()
            runtime.mkdir()
            for name in UTILITIES:
                tool = shutil.which(name, path="/usr/bin:/bin")
                if tool is None:
                    raise RuntimeError(f"Required host bootstrap utility is unavailable: {name}")
                (commands / name).symlink_to(tool)
            if shutil.which("ape", path=str(commands)) is not None:
                raise RuntimeError("An external APE loader is present on the restricted PATH")
            report["preinstalled_ape_on_path"] = False
            application = root / APPLICATION
            shutil.copyfile(source, application)
            application.chmod(0o755)
            if sha256(application) != original:
                raise RuntimeError("Bootstrap application copy differs from the verified artifact")
            if list(runtime.iterdir()):
                raise RuntimeError("Bootstrap TMPDIR is not empty before execution")
            report["initial_runtime_directory_empty"] = True
            report["initial_root_entries"] = sorted(path.name for path in root.iterdir())
            environment = {"PATH": str(commands), "TMPDIR": str(runtime),
                           "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "TERM": "dumb"}
            started = time.monotonic()
            try:
                with log.open("wb") as output:
                    result = subprocess.run(["/bin/sh", str(application), "--self-test"],
                                            cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                            stdout=output, stderr=subprocess.STDOUT, timeout=240)
            finally:
                report["elapsed_seconds"] = round(time.monotonic() - started, 3)
                report["sha256_copy_after"] = sha256(application)
                report["sha256_after"] = sha256(source)
                report["hash_unchanged"] = (report["sha256_copy_after"] == original
                                             == report["sha256_after"])
            report["exit_code"] = result.returncode
            report["command"] = ["/bin/sh", APPLICATION, "--self-test"]
            if result.returncode != 0:
                raise RuntimeError(f"Shell bootstrap self-test exited with code {result.returncode}")
            if not report["hash_unchanged"]:
                raise RuntimeError("Application bytes changed during shell bootstrap")
            report["greedy_token_ids"] = validate_selftest_output(log.read_text(errors="replace"))
            report["independent_reference_match"] = True
            report["bundled_loader"] = extracted_loader(runtime)
            report["runtime_entries_after"] = sorted(path.name for path in runtime.iterdir())
        report["temporary_directory_removed"] = True
        report["status"] = "passed"
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        report["error"] = str(error)
        if log.is_file():
            print(log.read_text(errors="replace")[-18000:], file=sys.stderr)
        print(f"Linux shell bootstrap verification failed: {error}", file=sys.stderr)
        return 1
    finally:
        (logs / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    return verify(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
