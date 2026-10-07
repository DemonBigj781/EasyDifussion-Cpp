#!/usr/bin/env python3
"""Verify actual selected-device graphs and trained llama inference.

This host harness intentionally permits native driver dependencies. It is separate
from the embedded provider's isolated, zero-native-library deployment checks.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

from verify_runtime import (APPLICATION, LOADER, EXPECTED_GREEDY_IDS,
                            runtime_environment, sha256, verify_package)

COUNTERS = ("graphs", "submissions", "dispatches", "matmuls", "readbacks")
COUNTER_PATTERN = (r"graphs=(\d+) submissions=(\d+) dispatches=(\d+) "
                   r"matmuls=(\d+) readbacks=(\d+)")


def one(lines, prefix, pattern):
    matches = [line for line in lines if line.startswith(prefix)]
    match = re.fullmatch(pattern, matches[0]) if len(matches) == 1 else None
    if match is None:
        raise RuntimeError("Expected exactly one valid " + prefix + " record")
    return match


def counters(values, minimum_readbacks=1):
    result = dict(zip(COUNTERS, map(int, values)))
    if (len(result) != 5 or any(value <= 0 for value in result.values()) or
            result["readbacks"] < minimum_readbacks or result["matmuls"] > result["dispatches"]):
        raise RuntimeError("Missing actual graph/dispatch/submission/matmul/readback evidence")
    return result


def validate_graph(text, policy, require_hardware):
    lines = text.splitlines()
    match = one(lines, "WEBGPU_DEVICE_EXECUTION", r"WEBGPU_DEVICE_EXECUTION selector=(\S+) "
                r"provider=(native|embedded) software=([01]) adapter_type=(\d+) "
                r"device_kind=(discrete-gpu|integrated-gpu|software-cpu|unknown) require_hardware=([01]) " +
                COUNTER_PATTERN + r" native_loader_opens=(\d+) cpu_fallback=0")
    selector, provider, software, adapter_type, kind, required = match.groups()[:6]
    software, adapter_type = int(software), int(adapter_type)
    hardware = provider == "native" and not software and adapter_type in (1, 2)
    expected_kind = ("software-cpu" if software else "discrete-gpu" if adapter_type == 1 else
                     "integrated-gpu" if adapter_type == 2 else "unknown")
    if (kind != expected_kind or int(required) != int(require_hardware) or
            (require_hardware and not hardware) or (policy != "auto" and provider != policy)):
        raise RuntimeError("Selected adapter classification/provider does not match the requested gate")
    opens = int(match.group(12))
    if (provider == "native" and opens <= 0) or (policy == "embedded" and opens != 0):
        raise RuntimeError("Native loader evidence disagrees with the selected provider")
    total = counters(match.groups()[6:11], 12)
    cases = {}
    for line in (line for line in lines if line.startswith("WEBGPU_GGML_EXECUTION")):
        case = re.fullmatch(r"WEBGPU_GGML_EXECUTION type=(F32|Q4_0) iteration=([01]) "
                           r"provider=(native|embedded) software=([01]) " + COUNTER_PATTERN +
                           r" native_loader_opens=(\d+) cpu_fallback=0", line)
        if not case or case.group(3) != provider or int(case.group(4)) != software:
            raise RuntimeError("Invalid or wrong-device graph execution record")
        key = (case.group(1), int(case.group(2)))
        if key in cases or int(case.group(10)) != opens:
            raise RuntimeError("Duplicate graph case or inconsistent native-loader count")
        cases[key] = {"type": key[0], "iteration": key[1], **counters(case.groups()[4:9], 3)}
    expected = {(kind, iteration) for kind in ("F32", "Q4_0") for iteration in (0, 1)}
    if set(cases) != expected:
        raise RuntimeError("Missing selected-device F32/Q4_0 graph cases")
    checks = {}
    for line in (line for line in lines if line.startswith("WEBGPU_CHECK")):
        check = re.fullmatch(r"WEBGPU_CHECK type=(F32|Q4_0) iteration=([01]) "
                            r"stage=(matmul|bias-rmsnorm-silu|softmax) values=143 "
                            r"max_error=([0-9.eE+-]+) PASS", line)
        if not check:
            raise RuntimeError("Malformed scalar-reference comparison")
        key = (check.group(1), int(check.group(2)), check.group(3))
        error = float(check.group(4))
        # The independent C oracle applies each stage's relative tolerance to
        # every element. Its reported maximum is an absolute error, not the
        # normalized error used by that per-element decision.
        if key in checks or not math.isfinite(error) or error < 0:
            raise RuntimeError("Duplicate or invalid numerical comparison")
        checks[key] = {"type": key[0], "iteration": key[1], "stage": key[2],
                       "values": 143, "max_error": error}
    if set(checks) != {(kind, iteration, stage) for kind, iteration in expected
                      for stage in ("matmul", "bias-rmsnorm-silu", "softmax")}:
        raise RuntimeError("Missing the twelve independent scalar readback comparisons")
    if [line for line in lines if line.startswith("WEBGPU_UNSUPPORTED")] != [
            "WEBGPU_UNSUPPORTED op=SILU_BACK supported=0"]:
        raise RuntimeError("Missing unsupported-operator rejection")
    if lines.count("WEBGPU_DEVICE_TEST PASS") != 1 or any("FAIL" in line for line in lines):
        raise RuntimeError("The device graph probe did not pass uniquely")
    adapter = one(lines, "WEBGPU_DEVICE_ADAPTER", r"WEBGPU_DEVICE_ADAPTER (.+)").group(1)
    return {"selector": selector, "provider": provider, "software": bool(software),
            "adapter_type": adapter_type, "device_kind": kind, "hardware_verified": hardware,
            "adapter_name": adapter, "native_loader_opens": opens, "cpu_fallback": False,
            "execution": total, "graph_cases": [cases[key] for key in sorted(cases)],
            "scalar_checks": [checks[key] for key in sorted(checks)]}


def validate_all(text, policy):
    lines = text.splitlines()
    count = int(one(lines, "WEBGPU_DEVICE_ALL", r"WEBGPU_DEVICE_ALL count=(\d+) PASS").group(1))
    blocks, current, selector = [], None, None
    for line in lines:
        if line.startswith("WEBGPU_DEVICE_BEGIN"):
            match = re.fullmatch(r"WEBGPU_DEVICE_BEGIN selector=(\S+)", line)
            if current is not None or not match:
                raise RuntimeError("Invalid all-adapter block boundary")
            selector, current = match.group(1), []
        elif line.startswith("WEBGPU_DEVICE_END"):
            match = re.fullmatch(r"WEBGPU_DEVICE_END selector=(\S+) result=0", line)
            if current is None or not match or match.group(1) != selector:
                raise RuntimeError("Incomplete all-adapter execution")
            result = validate_graph("\n".join(current), policy, False)
            if result["selector"] != selector:
                raise RuntimeError("Graph result belongs to a different adapter block")
            blocks.append(result)
            current = None
        elif current is not None:
            current.append(line)
        elif line.startswith(("WEBGPU_CHECK", "WEBGPU_GGML_EXECUTION", "WEBGPU_DEVICE_EXECUTION")):
            raise RuntimeError("Unscoped graph result outside an adapter block")
    if current is not None or not count or len(blocks) != count or len({b["selector"] for b in blocks}) != count:
        raise RuntimeError("All-adapter graph count or unique selector check failed")
    return blocks


def validate_llama(stdout, stderr, device):
    lines = stderr.splitlines()
    match = one(lines, "LLAMA_GENERATED_IDS", r"LLAMA_GENERATED_IDS ([0-9]+(?:,[0-9]+)*)")
    ids = list(map(int, match.group(1).split(",")))
    if ids != EXPECTED_GREEDY_IDS:
        raise RuntimeError("Selected-device llama differs from the independent sixteen-token reference")
    if lines.count("LLAMA_GENERATION prompt_tokens=5 decode_steps=16 vocab=512 finite_logits=1 threads=1") != 1:
        raise RuntimeError("Llama did not complete the sixteen finite-logit decode steps")
    selected = one(lines, "LLAMA_DEVICE", r"LLAMA_DEVICE selector=(\S+) provider=(native|embedded) software=([01]) stable_id=(\S+)")
    if selected.groups()[:3] != (device["selector"], device["provider"], str(int(device["software"]))):
        raise RuntimeError("Llama selected a different device")
    execution = one(lines, "WEBGPU_LLAMA_EXECUTION", r"WEBGPU_LLAMA_EXECUTION provider=(native|embedded) "
                    r"software=([01]) " + COUNTER_PATTERN + r" native_loader_opens=(\d+)")
    if execution.group(1) != device["provider"] or int(execution.group(2)) != device["software"]:
        raise RuntimeError("Llama dispatch provider/classification mismatch")
    opens = int(execution.group(8))
    if device["provider"] == "native" and opens <= 0:
        raise RuntimeError("Native llama did not open the native Vulkan loader")
    total = counters(execution.groups()[2:7])
    decode = one(lines, "WEBGPU_LLAMA_DECODE_EXECUTION", r"WEBGPU_LLAMA_DECODE_EXECUTION steps=16 " + COUNTER_PATTERN)
    scoped = counters(decode.groups())
    if any(scoped[name] > total[name] for name in COUNTERS):
        raise RuntimeError("Decode count exceeds total inference count")
    if not stdout.startswith("Once upon a time") or not stdout.strip():
        raise RuntimeError("Llama CLI did not emit generated text")
    return {"greedy_ids": ids, "independent_reference_match": True, "finite_logits": True,
            "decode_steps": 16, "execution": total, "decode_execution": scoped,
            "native_loader_opens": opens, "cpu_participation_allowed": True,
            "generated_text": stdout.strip()}


def verify(args):
    artifact, output = args.artifact_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise RuntimeError("Use a fresh output directory")
    app = artifact / APPLICATION
    report = {"status": "failed", "platform": platform.platform(), "provider_policy": args.provider,
              "requested_device": args.device, "all_adapters_same_process": args.all,
              "require_hardware": args.require_hardware, "filesystem_isolated": False,
              "software_driver_workers": 2, "commands": [], "devices": []}
    try:
        report["sha256_before"] = verify_package(artifact, args.expected_sha256 or os.environ.get("EXPECTED_SHA256"))
        base = [str(app)] if os.name == "nt" else [str(artifact / LOADER), str(app)]
        with tempfile.TemporaryDirectory(prefix="device-runtime-", dir=output) as temporary:
            temporary = Path(temporary)
            environment = runtime_environment(temporary)
            environment["LP_NUM_THREADS"] = "2"
            # The SDK's Linux cosmo_dlopen bridge builds its native host helper.
            # This native-driver check intentionally supplies a host compiler;
            # the separate embedded-provider isolation checks still omit it.
            if platform.system() == "Linux" and args.provider != "embedded":
                environment["PATH"] = "/usr/bin:/bin"
            if args.vulkan_icd:
                icd = args.vulkan_icd.resolve()
                if not icd.is_file():
                    raise RuntimeError("Vulkan ICD JSON does not exist")
                environment.update(VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd))
                report["vulkan_icd"] = {"path": str(icd), "sha256": sha256(icd)}
            common = ["--config", str(temporary / "config.json"), "--backend", "webgpu",
                      "--provider", args.provider]

            def run(label, arguments, expected=0):
                command = base + arguments
                record = {"label": label, "command": command, "expected_application_exit_code": expected,
                          "expected_host_exit_status": expected << 8 if os.name == "nt" else expected,
                          "stdout_log": label + ".stdout.log", "stderr_log": label + ".stderr.log"}
                report["commands"].append(record)
                start = time.monotonic()
                with (output / record["stdout_log"]).open("wb") as stdout, (output / record["stderr_log"]).open("wb") as stderr:
                    result = subprocess.run(command, cwd=temporary, env=environment, stdin=subprocess.DEVNULL,
                                            stdout=stdout, stderr=stderr, timeout=args.timeout)
                record.update(exit_code=result.returncode, elapsed_seconds=round(time.monotonic() - start, 3))
                if result.returncode != record["expected_host_exit_status"]:
                    raise RuntimeError(f"{label}: native status {result.returncode}, expected {record['expected_host_exit_status']}")
                return ((output / record["stdout_log"]).read_text(errors="replace"),
                        (output / record["stderr_log"]).read_text(errors="replace"))

            flags = ["--all"] if args.all else (["--require-hardware"] if args.require_hardware else [])
            stdout, stderr = run("graph", ["webgpu-device-test", *common, "--device", args.device, *flags])
            lane = one(stdout.splitlines(), "WEBGPU_DEVICE_LANE", r"WEBGPU_DEVICE_LANE mode=(direct|main-thread-service)").group(1)
            expected_lane = "main-thread-service" if platform.system() == "Linux" and args.provider != "embedded" else "direct"
            if lane != expected_lane:
                raise RuntimeError("Native main-thread service lane was not exercised")
            report["execution_lane"] = lane
            report["devices"] = (validate_all(stdout, args.provider) if args.all else
                                 [validate_graph(stdout, args.provider, args.require_hardware)])
            if args.all and args.provider == "auto" and args.vulkan_icd:
                if {device["provider"] for device in report["devices"]} != {"native", "embedded"}:
                    raise RuntimeError("Auto coexistence gate requires actual native and embedded graph executions")
                report["native_embedded_coexistence_verified"] = True
            for i, device in enumerate(report["devices"]):
                selected = [*common, "--device", device["selector"]]
                stdout, stderr = run(f"llama-{i}", ["llama", *selected, "--prompt", "Once upon a time",
                                     "--tokens", "16", "--threads", "1", "--report-tokens"])
                device["llama"] = validate_llama(stdout, stderr, device)
                if not device["hardware_verified"]:
                    stdout, stderr = run(f"hardware-rejection-{i}", ["webgpu-device-test", *selected,
                                                                   "--require-hardware"], expected=1)
                    expected = (f"WEBGPU_DEVICE_TEST FAIL: hardware required, selected {device['selector']} "
                                f"provider={device['provider']} type={device['device_kind']}")
                    if stderr.splitlines().count(expected) != 1 or "WEBGPU_DEVICE_EXECUTION" in stdout or "WEBGPU_DEVICE_TEST PASS" in stdout:
                        raise RuntimeError("Software/unknown hardware-required gate did not reject explicitly")
                    device["hardware_required_rejection_verified"] = True
                else:
                    device["hardware_required_rejection_verified"] = None
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
    finally:
        report["sha256_after"] = sha256(app) if app.is_file() else None
        if report.get("sha256_before") != report["sha256_after"]:
            report.update(status="failed", error="Application identity changed or was not established")
        report["evidence"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                              for path in sorted(output.glob("*.log"))}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("auto", "native", "embedded"), default="native")
    parser.add_argument("--device", default="WebGPU0")
    parser.add_argument("--vulkan-icd", type=Path)
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.all and args.require_hardware:
        parser.error("--all and --require-hardware are mutually exclusive")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.timeout > 1800:
        parser.error("--timeout must be greater than zero and no more than 1800 seconds")
    report = verify(args)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
