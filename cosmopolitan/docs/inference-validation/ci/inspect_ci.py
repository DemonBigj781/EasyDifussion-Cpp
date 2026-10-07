#!/usr/bin/env python3
"""Read-only final-CI evidence inspection; never launch the application.

Usage: inspect_ci.py --application-dir APP --linux-dir LINUX --windows-dir WINDOWS
APP contains the downloaded BUILD.json, SYMBOLS.json, PIN.json and packaged
verify_runtime.py, verify_inference.py, verify_inference_api.py. The executable
is optional: its absence is explicitly reported, never called a binary rehash.
Result directories contain the workflow's named results-*/report.json folders.
The two hash-pinned verifier modules are imported only for their pure saved-log
and PNG parsers. No subprocess, network, model loading or file writes are used.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
COMMIT = "6898a4e564c3363d1692f0aed1a2da0395a17a77"
PIN_SHA = "2c9870678a913bab2edc949c5acb9b4d69025a2cfaa6678ce3e8a89c321de0ca"
MODEL_SHA = "c2f6e92f9d08d69cc673a1003528ac8199274b3c0eaec88d5fbefe5af67bd42b"
VERIFIERS = {
    "verify_runtime": "1a15ba2f93237f946edb7ad68b34df5f72758a86e07af8d8a49485a015e5919a",
    "verify_inference": "feb52d0b0835e6cd6e9f8f05db7920d07e80f24e20872874e0020c02159ae036",
    "verify_inference_api": "cdb454477d6e90755e2c052c8753c8f7e906099fb9d70f32b64c483aa13d651b",
}
PATCHES = {
    "app-patches/0005-native-inference-worker-stack.patch": "c12c73c3702dbb5991d29b88ef35bc5c6243ecb0dd9a0feef2a846600fbe581d",
    "app-patches/0006-crow-async-response-lifetime.patch": "8b587738aa4dc0225eb3b5badac581360bef2f43982e8693b9cb539bef006a22",
    "app-patches/0007-native-inference-async-http.patch": "da404273407cb5dd1918eb588fe413af5aa02dec049582a86217d16ba2944d29",
}
COUNTERS = ("graphs", "submissions", "dispatches", "matmuls", "readbacks")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    require(type(actual) is type(expected) and actual == expected,
            f"{label}: expected {expected!r}, observed {actual!r}")


def finite(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f"{label}: invalid finite nonnegative number {value!r}")
    return value


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-JSON number {value} in {path}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                      object_pairs_hook=unique)


def saved_file(directory, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            f"Invalid evidence relative path: {relative!r}")
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()) and path.is_file(),
            f"Missing or escaping evidence file: {path}")
    return path


def load_parsers(directory):
    for name, expected in VERIFIERS.items():
        equal(digest(directory / (name + ".py")), expected, name + " source SHA256")
    modules = []
    for name in ("verify_runtime", "verify_inference"):
        spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def selftest_tokens(text, runtime):
    lines = text.splitlines()
    for marker in (*runtime.SELFTEST_MARKERS, "UI_SELFTEST pages=16 embedded_assets=yes PASS"):
        equal(lines.count(marker), 1, "Self-test success marker " + marker)
    records = [line for line in lines if line.startswith("LLAMA_GREEDY_IDS")]
    equal(len(records), 1, "CPU token record count")
    match = re.fullmatch(r"LLAMA_GREEDY_IDS ([0-9]+(?:,[0-9]+)*)", records[0])
    require(match is not None, "Malformed CPU token record")
    ids = list(map(int, match.group(1).split(",")))
    equal(ids, runtime.EXPECTED_GREEDY_IDS, "Independent CPU token oracle")
    return ids


def common_report(directory, os_name, app_sha):
    report = read_json(directory / "report.json")
    equal(report["status"], "passed", str(directory) + " status")
    require(report["platform"].startswith(os_name + "-"), f"Wrong report OS: {directory}")
    require("error" not in report, f"Report contains an error: {directory}")
    for key in ("sha256_before", "sha256_after"):
        equal(report[key], app_sha, str(directory) + " " + key)
    return report


def verify_evidence(directory, report, required):
    entries = report["evidence"]
    equal(len(entries), len(required), "Evidence file count")
    equal({entry["path"] for entry in entries}, set(required), "Evidence file names")
    for entry in entries:
        path = saved_file(directory, entry["path"])
        equal(path.stat().st_size, entry["bytes"], str(path) + " size")
        equal(digest(path), entry["sha256"], str(path) + " SHA256")


def model_integrity(report, pin, os_name):
    equal(report["model_pin"]["sha256"], PIN_SHA, "Model pin hash")
    equal(report["model_pin"]["metadata"], pin, "Model pin metadata")
    model = report["model"]
    for key in ("expected_sha256", "sha256_before", "sha256_after"):
        equal(model[key], MODEL_SHA, "Model " + key)
    equal(model["bytes"], pin["bytes"], "Model byte count")
    equal(report["app_hash_unchanged"], True, "Application unchanged")
    equal(report["model_hash_unchanged"], True, "Model unchanged")
    equal(report["filesystem_isolated"], False, "Model-test filesystem scope")
    equal(report["launch_mode"], "native Windows PE" if os_name == "Windows" else
          "explicit APE ELF loader", "Model-test launch mode")


def runtime_logs(directory, test):
    stdout = saved_file(directory, test["stdout_log"]).read_bytes()
    stderr = saved_file(directory, test["stderr_log"]).read_bytes()
    combined = saved_file(directory, test["log"]).read_bytes()
    expected = b"\n--- stdout ---\n" + stdout + b"\n--- stderr ---\n" + stderr
    require(combined == expected, "Runtime split/combined log bytes disagree: " + test["name"])
    text = combined.decode("utf-8", errors="replace")
    if test["name"] in {"devices", "diffusion-command", "image-command", "training-command"}:
        require((stdout + stderr).strip(), "Empty device/help output: " + test["name"])
    rejection = {
        "training-unmatched-option": "Missing value for native trainer option",
        "image-unmatched-option": "unknown, duplicate, or incomplete option",
        "image-invalid-dimensions": "invalid value for --width",
        "image-invalid-backend": "invalid value for --backend",
    }.get(test["name"])
    if rejection is not None:
        require(rejection in text, "Missing required rejection message: " + test["name"])
    return text


def runtime_report(directory, report, os_name, runtime):
    equal(report["isolated"], os_name == "Linux", "Runtime isolation")
    expected = {"self-test", "devices", "diffusion-command", "image-command", "training-command",
                "training-unmatched-option", "image-unmatched-option", "image-invalid-dimensions",
                "image-invalid-backend", "native-server-and-embedded-ui"}
    tests = report["tests"]
    equal(len(tests), len(expected), "Runtime test count")
    equal({test["name"] for test in tests}, expected, "Runtime test identities")
    for test in tests:
        text = runtime_logs(directory, test)
        if test["name"] != "native-server-and-embedded-ui":
            equal(test["exit_code"], test["expected_host_exit_status"], test["name"] + " exit")
            logical = 2 if test["name"] in {"training-unmatched-option", "image-unmatched-option",
                       "image-invalid-dimensions", "image-invalid-backend"} else 0
            equal(test["expected_application_exit_code"], logical, test["name"] + " logical exit")
            equal(test["expected_host_exit_status"], logical << 8 if os_name == "Windows" else logical,
                  test["name"] + " host exit")
        if test["name"] == "self-test":
            equal(test["greedy_token_ids"], selftest_tokens(text, runtime), "CPU reference tokens")
            equal(test["independent_reference_match"], True, "CPU reference match")
            equal(test["webgpu"], runtime.validate_webgpu_output(text), "WebGPU numerical/model logs")
            equal(test["inplace_storage"], runtime.validate_inplace_output(text), "In-place graph logs")
        if test["name"] == "native-server-and-embedded-ui":
            equal(test["ping_status"], 200, "Runtime HTTP ping")
            traversal = test["parent_traversal_status"]
            require(type(traversal) is int and traversal in (400, 404), "Runtime traversal was not rejected")
            paths = {"/", "/cpp-ui/training", "/cpp-ui/assets/ui.css", "/cpp-ui/scripts/generate.js",
                     "/cpp-ui/scripts/kiosk.js", "/v1/sdapi/v1/cosmopolitan-capabilities",
                     "/v1/sdapi/v1/checkpoints", "/v1/sdapi/v1/backend-devices"}
            equal(len(test["checks"]), len(paths), "Runtime HTTP route count")
            equal({check["path"] for check in test["checks"]}, paths, "Exact unique runtime HTTP routes")
            for check in test["checks"]:
                equal(check["status"], 200, "Runtime route " + check["path"])
                require(check["bytes"] > 0, "Empty runtime HTTP response")
    return {"tests": len(tests), "isolated": report["isolated"], "selftest_logs_reparsed": True}


def bootstrap_report(directory, report, runtime):
    for key, value in {"filesystem_isolated": False, "explicit_external_loader_supplied": False,
                       "preinstalled_ape_on_path": False, "initial_runtime_directory_empty": True,
                       "hash_unchanged": True, "temporary_directory_removed": True, "exit_code": 0,
                       "independent_reference_match": True}.items():
        equal(report[key], value, "Bootstrap " + key)
    equal(report["sha256_copy_after"], report["sha256_before"], "Bootstrap application copy hash")
    equal(report["command"], ["/bin/sh", "easy-diffusion.exe", "--self-test"], "Bootstrap command")
    loader = report["bundled_loader"]
    equal(loader["transient_extraction"], True, "Bootstrap transient extraction")
    equal(loader["format"], "x86-64 ELF", "Bootstrap loader format")
    equal(report["runtime_entries_after"], [loader["filename"]], "Bootstrap extracted files")
    text = saved_file(directory, report["log"]).read_text(errors="replace")
    equal(report["greedy_token_ids"], selftest_tokens(text, runtime), "Bootstrap CPU reference")
    equal(report["webgpu"], runtime.validate_webgpu_output(text), "Bootstrap WebGPU logs")
    return {"single_file_bootstrap": True, "transient_loader_extraction": True,
            "filesystem_isolated": False}


def cli_report(directory, report, os_name, backend, steps, pin, inference):
    model_integrity(report, pin, os_name)
    equal(report["backend"], backend, "Image backend")
    for key in ("exit_code", "expected_application_exit_code", "expected_host_exit_status"):
        equal(report[key], 0, "Image " + key)
    params = report["parameters"]
    for key, value in {"width": 256, "height": 256, "steps": steps, "seed": 42, "threads": 2,
                       "cfg_scale": 7.0}.items():
        equal(params[key], value, "Image parameter " + key)
    equal(report["software_driver_workers"], {"environment": "LP_NUM_THREADS", "value": 2},
          "Software driver thread limit")
    required = (report["stdout_log"], report["stderr_log"], report["output_png"])
    verify_evidence(directory, report, required)
    png = inference.inspect_png(saved_file(directory, report["output_png"]), 256, 256)
    equal(report["png"], png, "Decoded image evidence")
    args = SimpleNamespace(**params, backend=backend, sampler=None, scheduler=None)
    execution = inference.verify_markers(saved_file(directory, report["stdout_log"]).read_text(errors="replace"),
                                         saved_file(directory, report["stderr_log"]).read_text(errors="replace"),
                                         args, png)
    equal(report["execution"], execution, "Image execution report versus logs")
    # The parser requires every CPU counter to be zero and every WebGPU
    # generation/sampling counter to be positive, including actual readbacks.
    return {"backend": backend, "steps": steps, "elapsed_seconds": report["elapsed_seconds"],
            "generation": {key: execution["generation"][key] for key in COUNTERS},
            "sampling": {key: execution["sampling"][key] for key in COUNTERS},
            "png_sha256": png["sha256"], "quality_assessed": False}


def api_report(directory, report, os_name, pin, manifest, inference):
    model_integrity(report, pin, os_name)
    equal(report["backend"], "cpu", "API backend")
    equal(report["fresh_working_directory"], True, "API fresh working directory")
    equal(report["browser_execution_tested"], False, "API browser evidence scope")
    equal(report["server_exit_before_cleanup"], None, "API server alive before cleanup")
    # The pinned harness uses Popen.terminate(): CPython calls TerminateProcess
    # with status 1 on Windows. Reject a crash racing the preceding poll(None),
    # rather than accepting an arbitrary native status as intended cleanup.
    equal(report["server_cleanup_exit_status"], 1 if os_name == "Windows" else 0,
          "API expected native cleanup status")
    verify_evidence(directory, report, ("server.stdout.log", "server.stderr.log", "api-image.png"))
    workers = report["inference_workers"]
    equal(len(workers), 2, "API inference worker count")
    for worker in workers:
        equal(worker["stack_bytes"], 8388608, "Measured API worker stack")
        equal(worker["source"], "pthread_getattr_np", "Worker measurement source")
        require(type(worker["guard_bytes"]) is int and worker["guard_bytes"] > 0, "Invalid worker guard")
    matches = re.findall(r"^NATIVE_INFERENCE_WORKER stack_bytes=(\d+) guard_bytes=(\d+) "
                         r"stack_source=pthread_getattr_np$",
                         saved_file(directory, "server.stderr.log").read_text(errors="replace"), re.MULTILINE)
    equal(workers, [{"stack_bytes": int(size), "guard_bytes": int(guard), "source": "pthread_getattr_np"}
                    for size, guard in matches], "API worker report versus logs")
    generation = report["generation"]
    params = generation["parameters"]
    for key, value in {"width": 256, "height": 256, "steps": 4, "seed": 42, "backend": "cpu",
                       "batch_size": 1, "sampler_name": "euler", "scheduler": "discrete"}.items():
        equal(params[key], value, "API generation parameter " + key)
    equal(params["force_task_id"], generation["task_id"], "API generation identity")
    equal(params["override_settings"], {"sd_model_checkpoint": pin["filename"],
          "forge_additional_modules": [], "live_previews_enable": False}, "API request-only checkpoint")
    equal(generation["observed_active_progress"]["completed"], False, "API active observation")
    for key, value in {"completed": True, "interrupted": False, "current_step": 4, "total_steps": 4}.items():
        equal(generation["final_progress"][key], value, "API generation completion " + key)
    for key in ("concurrent_generation_status", "concurrent_options_status", "wrong_task_interrupt_status",
                "finished_task_interrupt_status", "duplicate_task_id_status"):
        equal(generation[key], 409, "API " + key)
    equal(generation["persistent_options_unchanged"], True, "API persistent options")
    responsiveness = generation["active_progress_responsiveness"]
    equal(responsiveness["requests"], 32, "Active progress request count")
    equal(responsiveness["timeout_seconds"], 2, "Active progress socket timeout")
    times = responsiveness["elapsed_seconds"]
    equal(len(times), 32, "Recorded active progress durations")
    for value in times:
        finite(value, "Progress duration")
    equal(responsiveness["max_seconds"], max(times), "Maximum progress duration")
    # urllib's 2s timeout bounds socket operations, not total wall-clock time;
    # preserve measured durations rather than invent a stricter wall-time gate.
    equal(generation["png"], inference.inspect_png(saved_file(directory, "api-image.png"), 256, 256),
          "API decoded PNG evidence")
    cancel = report["cancellation"]
    equal(cancel["requested_steps"], 8, "Cancellation requested steps")
    equal(cancel["interrupt_status"], 200, "Matched task cancellation response")
    require(cancel["generation_status"] in (200, 500), "Unexpected canceled generation status")
    require(cancel["task_id"] != generation["task_id"], "Cancellation reused the finished task ID")
    for record in (cancel["sampling_observed"], cancel["final_progress"]):
        equal(record["total_steps"], 8, "Cancellation total steps")
        require(type(record["current_step"]) is int and 0 < record["current_step"] < 8,
                "Cancellation did not happen during sampling before all eight steps")
    equal(cancel["sampling_observed"]["completed"], False, "Cancellation active sampling")
    for key in ("completed", "interrupted"):
        equal(cancel["final_progress"][key], True, "Cancellation final " + key)
    for key in ("early_cancellation_observed", "gate_released"):
        equal(cancel[key], True, "Cancellation " + key)
    invalid = report["invalid_overrides"]
    equal(len(invalid), 3, "Invalid override cases")
    for case in invalid:
        equal(case["status"], 400, "Invalid override response")
        equal(case["gate_released"], True, "Invalid override gate release")
    ui = report["ui"]
    equal(ui["browser_execution_tested"], False, "Served-assets browser scope")
    equal(ui["capabilities"]["txt2img"], True, "Native txt2img capability")
    equal(ui["capabilities"]["max_concurrent_generations"], 1, "Native generation admission")
    equal(len(ui["scripts"]), 2, "Portable Generate script count")
    equal({entry["path"] for entry in ui["scripts"]},
          {"/cpp-ui/scripts/generate.js", "/cpp-ui/scripts/kiosk.js"}, "Served Generate scripts")
    for script in ui["scripts"]:
        resource = manifest["embedded_resources"][script["path"].lstrip("/")]
        equal({"bytes": script["bytes"], "sha256": script["sha256"]}, resource, "Served script manifest identity")
    return {"elapsed_seconds": report["elapsed_seconds"], "active_progress_requests": 32,
            "max_progress_seconds": max(times), "measured_worker_stacks": [worker["stack_bytes"] for worker in workers],
            "cancelled_at_step": cancel["final_progress"]["current_step"],
            "cancel_generation_status": cancel["generation_status"],
            "server_alive_before_cleanup": True, "cleanup_exit_status": report["server_cleanup_exit_status"],
            "png_sha256": generation["png"]["sha256"], "browser_execution_tested": False}


def inspect(args):
    app = args.application_dir.resolve()
    runtime, inference = load_parsers(app)
    manifest = read_json(app / "BUILD.json")
    equal(manifest["source"]["commit"], COMMIT, "Final source commit")
    equal(manifest["source"]["dirty"], False, "Clean source tree")
    equal(manifest["source"]["repository"], "DemonBigj781/EasyDifussion-Cpp", "Source repository")
    for name, expected in PATCHES.items():
        equal(manifest["build_recipe_sha256"][name], expected, name + " source identity")
    executable = manifest["executable"]
    equal(executable["filename"], "easy-diffusion.exe", "Application name")
    require(re.fullmatch(r"[0-9a-f]{64}", executable["sha256"]) is not None, "Invalid application SHA256")
    require(type(executable["bytes"]) is int and executable["bytes"] > 0, "Invalid executable size")
    present = (app / executable["filename"]).exists()
    if present:
        equal((app / executable["filename"]).stat().st_size, executable["bytes"], "Actual executable size")
        equal(runtime.verify_package(app, executable["sha256"]), executable["sha256"], "Actual package identity")
    equal(digest(app / "PIN.json"), PIN_SHA, "Packaged external model pin")
    pin = read_json(app / "PIN.json")
    equal(pin["sha256"], MODEL_SHA, "Pinned trained checkpoint")
    symbols = read_json(app / "SYMBOLS.json")
    equal(symbols["status"], "passed", "Symbol audit status")
    equal(symbols["legacy_versioned_ggml_symbols"], 0, "No renamed second GGML")
    equal(symbols["shared_ggml_webgpu_sources"], ["ggml-webgpu/ggml-webgpu.cpp"], "Single WebGPU GGML source")
    for symbol in ("ggml_init", "llama_model_load_from_file", "new_sd_ctx", "generate_image",
                   "cosmo_image_generate", "cosmo_sdkit_main", "ggml_backend_webgpu_init", "wgpuQueueSubmit",
                   "lvp_GetInstanceProcAddr", "LLVMCreateMCJITCompilerForModule"):
        equal(symbols["required_symbol_counts"][symbol], 1, "Required static symbol " + symbol)
    results = {}
    for os_name, base in (("Linux", args.linux_dir), ("Windows", args.windows_dir)):
        names = [("runtime", "results-linux-isolated" if os_name == "Linux" else "results-windows"),
                 ("cpu", "results-inference-cpu"), ("webgpu", "results-inference-webgpu"),
                 ("api", "results-inference-api")]
        if os_name == "Linux":
            names.append(("bootstrap", "results-linux-bootstrap"))
        for kind, name in names:
            directory = base / name
            report = common_report(directory, os_name, executable["sha256"])
            if kind == "runtime":
                detail = runtime_report(directory, report, os_name, runtime)
            elif kind == "bootstrap":
                detail = bootstrap_report(directory, report, runtime)
            elif kind == "api":
                detail = api_report(directory, report, os_name, pin, manifest, inference)
            else:
                detail = cli_report(directory, report, os_name, kind, 4 if kind == "cpu" else 2, pin, inference)
            results[os_name.lower() + "/" + kind] = {"report_sha256": digest(directory / "report.json"), **detail}
    equal(len(results), 9, "All nine report groups")
    return {"status": "passed", "source_commit": COMMIT, "source_dirty": False,
            "application": {**executable, "actual_binary_and_embedded_resources_rehashed": present},
            "model": {"sha256": MODEL_SHA, "bytes": pin["bytes"], "pin_sha256": PIN_SHA,
                      "actual_weights_rehashed_by_this_inspector": False,
                      "reported_before_after_hashes_verified": True},
            "report_groups_checked": 9, "reports": results,
            "scope": "Read-only saved-evidence inspection; no application execution, model loading or visual quality assessment"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-dir", type=Path, required=True)
    parser.add_argument("--linux-dir", type=Path, required=True)
    parser.add_argument("--windows-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = inspect(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
