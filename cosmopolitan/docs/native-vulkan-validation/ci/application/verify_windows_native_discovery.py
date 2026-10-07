#!/usr/bin/env python3
"""Exercise installed Chrome's native Vulkan DLLs and truthful f16 rejection.

No downloads or driver installation. Official source contracts:
https://github.com/actions/runner-images/blob/main/images/windows/Windows2022-Readme.md
https://github.com/chromium/chromium/blob/main/chrome/installer/mini_installer/chrome.release
https://github.com/google/swiftshader/blob/master/src/Vulkan/VkPhysicalDevice.cpp

Chrome distributes vulkan-1.dll, vk_swiftshader.dll and its ICD JSON together.
SwiftShader reports shaderFloat16=false; current GGML kernels require it.
This is native PE/DLL/ABI discovery and feature rejection, NOT GPU computation.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

from verify_runtime import APPLICATION, runtime_environment, sha256, verify_package

SOURCES = [
    "https://github.com/actions/runner-images/blob/main/images/windows/Windows2022-Readme.md",
    "https://github.com/chromium/chromium/blob/main/chrome/installer/mini_installer/chrome.release",
    "https://github.com/google/swiftshader/blob/master/src/Vulkan/VkPhysicalDevice.cpp",
]
REASON = "ShaderF16 is required by the compiled GGML kernels"


def chrome_directory():
    checked, candidates = [], []
    roots = {os.environ.get("ProgramFiles", r"C:\Program Files"),
             os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")}
    for root in sorted(roots):
        application = Path(root) / "Google/Chrome/Application"
        checked.append(str(application))
        if not application.is_dir():
            continue
        for version in application.iterdir():
            if not version.is_dir() or not re.fullmatch(r"\d+(?:\.\d+){3}", version.name):
                continue
            if all((version / name).is_file() for name in
                   ("vulkan-1.dll", "vk_swiftshader.dll", "vk_swiftshader_icd.json")):
                candidates.append(version)
    if not candidates:
        raise RuntimeError("Installed Chrome lacks the expected Vulkan loader/SwiftShader files; searched " + ", ".join(checked))
    return max(candidates, key=lambda path: tuple(map(int, path.name.split("."))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Use a fresh --output-dir")
    report = {"status": "failed", "scope": "native Windows PE and official installed Chrome Vulkan/SwiftShader DLL discovery with ShaderF16 rejection",
              "gpu_compute_tested": False, "physical_hardware_tested": False,
              "platform": platform.platform(), "primary_sources": SOURCES, "commands": []}
    started = time.monotonic()
    try:
        if os.name != "nt":
            raise RuntimeError("This gate requires native Windows; no skipped pass")
        directory = args.artifact_dir.resolve()
        original = directory / APPLICATION
        expected = args.expected_sha256 or os.environ.get("EXPECTED_SHA256")
        report["sha256_before"] = verify_package(directory, expected)
        chrome = chrome_directory()
        report["chrome_version_directory"] = str(chrome)
        report["chrome_version"] = chrome.name
        names = ("vulkan-1.dll", "vk_swiftshader.dll", "vk_swiftshader_icd.json")
        report["installed_inputs"] = [{"path": str(chrome / name), "bytes": (chrome / name).stat().st_size,
                                        "sha256_before": sha256(chrome / name)} for name in names]
        manifest = json.loads((chrome / "vk_swiftshader_icd.json").read_text())
        if not isinstance(manifest.get("ICD"), dict) or not isinstance(manifest["ICD"].get("api_version"), str):
            raise RuntimeError("Chrome's SwiftShader ICD manifest has an unsupported shape")
        manifest["ICD"]["library_path"] = str(chrome / "vk_swiftshader.dll")
        # Copy the application unchanged and the official loader next to it.
        # Windows resolves this application-directory DLL before system/PATH
        # copies. The ICD uses an absolute installed-driver path.
        with tempfile.TemporaryDirectory(prefix="chrome-vulkan-", dir=output) as temporary:
            work = Path(temporary)
            copied = work / APPLICATION
            shutil.copy2(original, copied)
            shutil.copy2(chrome / "vulkan-1.dll", work / "vulkan-1.dll")
            if sha256(copied) != report["sha256_before"]:
                raise RuntimeError("Application copy changed")
            if sha256(work / "vulkan-1.dll") != report["installed_inputs"][0]["sha256_before"]:
                raise RuntimeError("Native loader copy changed")
            icd = work / "swiftshader.json"
            icd.write_text(json.dumps(manifest, indent=2) + "\n")
            report["effective_icd"] = manifest
            environment = runtime_environment(work)
            environment.update({"VK_DRIVER_FILES": str(icd), "VK_ICD_FILENAMES": str(icd), "COSMO_WGPU_TRACE": "1"})
            for label, arguments, status in (
                ("discovery", ["devices", "--provider", "native", "--backend", "cpu"], 0),
                ("feature-rejection", ["webgpu-device-test", "--provider", "native", "--device", "auto"], 256),
            ):
                command = [str(copied), *arguments]
                begin = time.monotonic()
                result = subprocess.run(command, cwd=work, env=environment, stdin=subprocess.DEVNULL,
                                        capture_output=True, timeout=90)
                stdout, stderr = output / (label + ".stdout.log"), output / (label + ".stderr.log")
                stdout.write_bytes(result.stdout)
                stderr.write_bytes(result.stderr)
                report["commands"].append({"name": label, "command": command, "returncode": result.returncode,
                                           "expected_native_returncode": status,
                                           "expected_logical_application_code": 0 if status == 0 else 1,
                                           "elapsed_seconds": round(time.monotonic() - begin, 3)})
                text, errors = result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace")
                if result.returncode != status:
                    raise RuntimeError(f"{label}: native exit {result.returncode}, expected {status}; a crash is not feature rejection")
                if "WebGPU: Vulkan provider=native; library=vulkan-1.dll" not in errors:
                    raise RuntimeError(f"{label}: missing actual native-loader trace")
                if label == "discovery":
                    records = [line.split("\t") for line in text.splitlines() if line.startswith("unavailable\t")]
                    matches = [row for row in records if len(row) == 4 and row[1] == "native"
                               and "swiftshader" in row[2].lower() and row[3] == REASON]
                    if len(matches) != 1:
                        raise RuntimeError("Expected one real SwiftShader adapter rejected for ShaderF16; missing driver discovery is not a pass")
                    if any(re.match(r"^\d+\tWebGPU\t", line) for line in text.splitlines()):
                        raise RuntimeError("SwiftShader was advertised as usable despite the expected f16 limitation")
                    report["unavailable_adapter"] = {"provider": matches[0][1], "name": matches[0][2], "reason": matches[0][3]}
                elif ("WEBGPU_DEVICE_TEST FAIL:" not in errors or
                      "lacks required ShaderF16; skipped" not in errors or
                      "WEBGPU_DEVICE_TEST PASS" in text or "WEBGPU_DEVICE_EXECUTION" in text):
                    raise RuntimeError("Native compute was not rejected explicitly for the discovered unsupported adapter")
            report["copied_sha256_after"] = sha256(copied)
        report["sha256_after"] = sha256(original)
        if report["sha256_before"] != report["sha256_after"] or report["sha256_before"] != report["copied_sha256_after"]:
            raise RuntimeError("Application changed during discovery")
        for item in report["installed_inputs"]:
            item["sha256_after"] = sha256(Path(item["path"]))
            if item["sha256_before"] != item["sha256_after"]:
                raise RuntimeError("An installed Chrome input changed during discovery")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["evidence"] = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                          for path in sorted(output.glob("*.log"))]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
