#!/usr/bin/env python3
"""Verify native HTTP image inference, request ownership, and portable UI assets.

The model is an external, hash-pinned input. Python runs only as the host test
client. The server has a fresh working directory and a sanitized environment;
this verifier does not claim filesystem isolation or browser automation.
"""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

from verify_inference import inspect_png, read_model_pin
from verify_runtime import APPLICATION, LOADER, runtime_environment, sha256, verify_package


PREFIX = "/v1/sdapi/v1/"
MAX_RESPONSE = 64 * 1024 * 1024


class Client:
    def __init__(self, base, deadline, events):
        self.base, self.deadline, self.events = base, deadline, events

    def request(self, path, body=None, timeout=10, raw=False):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Native API verification deadline exceeded")
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data,
                                         headers={"Content-Type": "application/json"} if data is not None else {})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        started = time.monotonic()
        try:
            try:
                response = opener.open(request, timeout=min(timeout, remaining))
            except urllib.error.HTTPError as error:
                response = error
            with response:
                status = response.status
                payload = response.read(MAX_RESPONSE + 1)
                headers = dict(response.headers)
            if len(payload) > MAX_RESPONSE:
                raise RuntimeError(f"Oversized response for {path}")
            self.events.append({"path": path, "method": request.get_method(), "status": status,
                                "bytes": len(payload), "seconds": round(time.monotonic() - started, 4)})
            if raw:
                return status, payload, headers
            try:
                value = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                value = {"message": payload.decode(errors="replace")}
            return status, value, headers
        except Exception as error:
            self.events.append({"path": path, "method": request.get_method(), "error": str(error),
                                "seconds": round(time.monotonic() - started, 4)})
            raise

    def expect(self, path, body=None, status=200, timeout=10):
        actual, value, _ = self.request(path, body, timeout=timeout)
        if actual != status:
            raise RuntimeError(f"{path}: expected HTTP {status}, got {actual}: {str(value)[:500]}")
        return value


class PendingRequest:
    def __init__(self, client, body, timeout):
        self.done = threading.Event()
        self.response = None
        self.error = None

        def run():
            try:
                self.response = client.request(PREFIX + "txt2img", body, timeout=timeout)
            except Exception as error:
                self.error = error
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, name="native-txt2img-client", daemon=True)
        self.thread.start()

    def result(self, deadline):
        if not self.done.wait(max(0, deadline - time.monotonic())):
            raise TimeoutError("Native generation POST did not complete before the deadline")
        if self.error is not None:
            raise self.error
        return self.response


def task_body(model, steps, backend="cpu"):
    return {"force_task_id": "cosmo-api-" + uuid.uuid4().hex,
            "prompt": "a photograph of a red apple on a wooden table, natural light",
            "negative_prompt": "", "width": 256, "height": 256, "steps": steps,
            "cfg_scale": 7.0, "seed": 42, "batch_size": 1, "backend": backend,
            "sampler_name": "euler", "scheduler": "discrete",
            "override_settings": {"sd_model_checkpoint": model, "forge_additional_modules": [],
                                  "live_previews_enable": False}}


def progress(client, task_id, timeout=10):
    status, value, _ = client.request("/v1/internal/progress", {"id_task": task_id, "live_preview": False},
                                      timeout=timeout)
    if status == 404:
        return None
    if status != 200 or not isinstance(value, dict):
        raise RuntimeError(f"Progress request failed: HTTP {status}: {value}")
    if type(value.get("completed")) is not bool:
        raise RuntimeError("Progress response has no boolean completion state")
    for name in ("current_step", "total_steps"):
        if type(value.get(name)) not in (int, float) or not math.isfinite(value[name]) or value[name] < 0:
            raise RuntimeError(f"Invalid progress {name}")
    return value


def wait_for_task(client, request, task_id, process, deadline, sampling=False):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Native server exited while a generation request was pending")
        state = progress(client, task_id)
        if state is not None and not state["completed"] and (not sampling or state["current_step"] > 0):
            return state
        if request.done.is_set():
            status, body, _ = request.result(deadline)
            raise RuntimeError(f"Request finished before the {'sampling' if sampling else 'active'} observation window: "
                               f"HTTP {status}; image count {len(body.get('images', [])) if isinstance(body, dict) else 'unknown'}")
        time.sleep(0.05 if sampling else 0.1)
    raise TimeoutError("Timed out waiting for native task progress")


def inspect_ui(client, manifest):
    capabilities = client.expect(PREFIX + "cosmopolitan-capabilities")
    required = {"protocol": 1, "mode": "native-single-user", "kiosk_supported": False,
                "kiosk_enabled": False, "txt2img": True, "max_concurrent_generations": 1,
                "cancel_during_model_load": False}
    if any(type(capabilities.get(key)) is not type(value) or capabilities[key] != value
           for key, value in required.items()):
        raise RuntimeError("Native mode/capabilities differ from the portable UI contract")
    policy = client.expect("/kiosk")
    if policy.get("supported") is not False or policy.get("enabled") is not False:
        raise RuntimeError("Native kiosk mode was not explicitly unavailable/disabled")
    client.expect("/kiosk", {"enabled": True}, status=501)
    if client.expect("/kiosk") != policy:
        raise RuntimeError("Unsupported kiosk mutation changed the reported mode")
    status, html, headers = client.request("/", raw=True)
    if status != 200 or "text/html" not in headers.get("Content-Type", ""):
        raise RuntimeError("Portable Generate page is unavailable")
    scripts = re.findall(rb'<script\s+[^>]*src="([^"]+)"', html)
    expected = [b"/cpp-ui/scripts/kiosk.js", b"/cpp-ui/scripts/generate.js"]
    if scripts != expected or b'id="makeImage"' not in html or b'id="stable_diffusion_model"' not in html:
        raise RuntimeError("Generate page is missing its DOM or still loads unsupported application scripts")
    assets = []
    for route in expected:
        path = route.decode()
        status, contents, headers = client.request(path, raw=True)
        name = path.lstrip("/")
        stored = manifest["embedded_resources"][name]
        digest = hashlib.sha256(contents).hexdigest()
        if status != 200 or "javascript" not in headers.get("Content-Type", "") or digest != stored["sha256"]:
            raise RuntimeError(f"Served UI asset differs from packaged resource: {path}")
        if path.endswith("/generate.js") and (b'"/render"' in contents or b"override_settings" not in contents
                                               or b"/v1/sdapi/v1/txt2img" not in contents):
            raise RuntimeError("Generate script is not the native inference adapter")
        assets.append({"path": path, "bytes": len(contents), "sha256": digest})
    return {"capabilities": capabilities, "kiosk": policy, "scripts": assets,
            "browser_execution_tested": False}


def generation_checks(client, model, baseline, process, deadline, logs, requests, steps, backend):
    body = task_body(model, steps, backend)
    pending = PendingRequest(client, body, deadline - time.monotonic())
    requests.append(pending)
    active = wait_for_task(client, pending, body["force_task_id"], process, deadline)
    probe = task_body(model, 4)
    # If the first request has just finished, this invalid model selector avoids
    # accidentally starting another expensive generation during the busy probe.
    probe["override_settings"]["sd_model_checkpoint"] = "__missing_busy_probe__.gguf"
    for path, payload in ((PREFIX + "txt2img", probe), (PREFIX + "options", baseline)):
        if pending.done.is_set():
            raise RuntimeError("Busy-request observation window was missed: generation already finished")
        status, value, _ = client.request(path, payload)
        if status != 409:
            if pending.done.is_set():
                raise RuntimeError("Busy-request observation window was missed during the HTTP probe; "
                                   "no concurrent-request rejection success is claimed")
            raise RuntimeError(f"Active generation did not reject {path}: HTTP {status}: {value}")
    client.expect(PREFIX + "interrupt", {"id_task": "wrong-task-" + uuid.uuid4().hex}, status=409)
    if client.expect(PREFIX + "options") != baseline:
        raise RuntimeError("Request-local checkpoint options leaked into persistent GET options")
    # Each client call opens a new connection. Repeated requests expose a Crow
    # I/O worker blocked by a synchronous generation handler, even when the
    # first few progress/409 responses happened to use available workers.
    progress_seconds = []
    for _ in range(32):
        if pending.done.is_set():
            raise RuntimeError("Active progress responsiveness observation window was missed")
        started = time.monotonic()
        state = progress(client, body["force_task_id"], timeout=2)
        progress_seconds.append(round(time.monotonic() - started, 6))
        if state is None or state["completed"]:
            raise RuntimeError("Generation was not active throughout the progress responsiveness probe")
    status, result, _ = pending.result(deadline)
    if status != 200 or not isinstance(result, dict) or len(result.get("images", [])) != 1:
        raise RuntimeError(f"Real native txt2img did not produce one image: HTTP {status}: {str(result)[:500]}")
    if result.get("task_id") != body["force_task_id"]:
        raise RuntimeError("Generation response did not identify its requested task")
    encoded = result["images"][0]
    if not isinstance(encoded, str):
        raise RuntimeError("Image result is not a base64 string")
    image = logs / "api-image.png"
    image.write_bytes(base64.b64decode(encoded, validate=True))
    decoded = inspect_png(image, 256, 256)
    state = progress(client, body["force_task_id"])
    if state is None or not state["completed"]:
        raise RuntimeError("Successful generation task is not marked complete")
    if client.expect(PREFIX + "options") != baseline:
        raise RuntimeError("Generation override changed persistent options")
    client.expect(PREFIX + "interrupt", {"id_task": body["force_task_id"]}, status=409)
    client.expect(PREFIX + "txt2img", body, status=409)
    client.expect(PREFIX + "options", baseline)
    return {"task_id": body["force_task_id"], "parameters": body, "observed_active_progress": active,
            "final_progress": state, "png": decoded, "concurrent_generation_status": 409,
            "concurrent_options_status": 409, "wrong_task_interrupt_status": 409,
            "finished_task_interrupt_status": 409, "duplicate_task_id_status": 409,
            "persistent_options_unchanged": True,
            "active_progress_responsiveness": {"requests": len(progress_seconds), "timeout_seconds": 2,
                                               "elapsed_seconds": progress_seconds,
                                               "max_seconds": max(progress_seconds)}}


def invalid_override_checks(client, model, baseline):
    tests = []
    for override in ({"sd_model_checkpoint": 42}, {"unknown_native_option": True},
                     {"sd_model_checkpoint": "__missing_indexed_checkpoint__.gguf"}):
        body = task_body(model, 4)
        body["override_settings"] = override
        error = client.expect(PREFIX + "txt2img", body, status=400)
        if not isinstance(error, dict) or not isinstance(error.get("message"), str) or not error["message"]:
            raise RuntimeError("Invalid override did not return a useful error message")
        client.expect(PREFIX + "options", baseline)
        if client.expect(PREFIX + "options") != baseline:
            raise RuntimeError("Invalid override changed persistent options")
        tests.append({"override": override, "status": 400, "message": error["message"], "gate_released": True})
    return tests


def cancellation_check(client, model, baseline, process, deadline, requests, steps, backend):
    body = task_body(model, steps, backend)
    pending = PendingRequest(client, body, deadline - time.monotonic())
    requests.append(pending)
    observed = wait_for_task(client, pending, body["force_task_id"], process, deadline, sampling=True)
    status, result, _ = client.request(PREFIX + "interrupt", {"id_task": body["force_task_id"]})
    if status != 200:
        state = progress(client, body["force_task_id"])
        if state and state["completed"]:
            raise RuntimeError("Cancellation observation window was missed: sampling completed before interrupt; "
                               "no cancellation success is claimed")
        raise RuntimeError(f"Active task cancellation failed: HTTP {status}: {result}")
    status, result, _ = pending.result(deadline)
    state = progress(client, body["force_task_id"])
    if not state or not state["completed"] or state.get("interrupted") is not True:
        raise RuntimeError("Cancelled request did not complete with its interruption recorded")
    if state["current_step"] >= steps:
        raise RuntimeError("Interrupt was accepted after all sampling steps; early cancellation was not observed")
    if status == 200:
        if not isinstance(result, dict) or result.get("images") != []:
            raise RuntimeError("Cancelled request unexpectedly returned completed images")
    elif status == 500:
        if not isinstance(result, dict) or not isinstance(result.get("message"), str) or not result["message"]:
            raise RuntimeError("Cancelled request returned an uninformative native error")
    else:
        raise RuntimeError(f"Unexpected cancelled generation HTTP status {status}")
    client.expect(PREFIX + "interrupt", {"id_task": body["force_task_id"]}, status=409)
    client.expect(PREFIX + "options", baseline)
    if client.expect(PREFIX + "options") != baseline:
        raise RuntimeError("Cancelled request leaked its option overrides")
    return {"task_id": body["force_task_id"], "requested_steps": steps, "sampling_observed": observed,
            "interrupt_status": 200, "generation_status": status, "final_progress": state,
            "early_cancellation_observed": True, "gate_released": True}


def verify(args):
    directory, model, logs = args.artifact_dir.resolve(), args.model.resolve(), args.output_dir.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    if any(logs.iterdir()):
        raise RuntimeError("Use a fresh --output-dir so old image/log evidence cannot satisfy this run")
    report = {"status": "failed", "platform": platform.platform(), "backend": args.backend,
              "provider_policy": args.provider, "requested_device": args.device,
              "require_hardware": args.require_hardware,
              "scope": "Real pinned SD1.5 native HTTP inference, request lifecycle, and served Generate assets",
              "filesystem_isolated": False, "fresh_working_directory": True,
              "browser_execution_tested": False, "model": {}, "http_events": []}
    started = time.monotonic()
    deadline = started + args.timeout
    process, requests = None, []
    app = directory / APPLICATION
    try:
        pin, report["model_pin"] = read_model_pin(args)
        if not model.is_file() or model.stat().st_size != pin["bytes"]:
            raise RuntimeError("Model file size does not match PIN.json")
        report["model"] = {"path": str(model), "expected_sha256": pin["sha256"], "bytes": pin["bytes"],
                           "sha256_before": sha256(model)}
        if report["model"]["sha256_before"] != pin["sha256"]:
            raise RuntimeError("Model SHA256 does not match PIN.json")
        report["sha256_before"] = verify_package(directory, args.expected_sha256 or os.environ.get("EXPECTED_SHA256"))
        manifest = json.loads((directory / "BUILD.json").read_text())
        with socket.socket() as socket_:
            socket_.bind(("127.0.0.1", 0))
            port = socket_.getsockname()[1]
        command = [str(app)] if os.name == "nt" else [str(directory / LOADER), str(app)]
        command += ["sdkit", "--backend", args.backend, "--provider", args.provider,
                    "--device", args.device, "--ckpt-dir", str(model.parent), "--port", str(port)]
        report["command"] = command
        report["launch_mode"] = "native Windows PE" if os.name == "nt" else "explicit APE ELF loader"
        client = Client(f"http://127.0.0.1:{port}", deadline, report["http_events"])
        with tempfile.TemporaryDirectory(prefix="native-api-", dir=logs) as temporary:
            work = Path(temporary)
            environment = runtime_environment(work)
            if args.provider != "embedded" and os.name != "nt":
                # Cosmopolitan 4.0.2 builds its native dlopen helper using host cc.
                # This run intentionally permits host libraries and a compiler.
                environment["PATH"] = "/usr/bin:/bin"
            if args.vulkan_icd:
                environment["VK_DRIVER_FILES"] = str(args.vulkan_icd.resolve())
                report["vulkan_icd"] = str(args.vulkan_icd.resolve())
            environment["LP_NUM_THREADS"] = str(args.lavapipe_threads)
            report["lavapipe_threads"] = args.lavapipe_threads
            stdout, stderr = logs / "server.stdout.log", logs / "server.stderr.log"
            with stdout.open("wb") as output, stderr.open("wb") as errors:
                process = subprocess.Popen(command, cwd=work, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=output, stderr=errors)
                try:
                    startup_deadline = min(deadline, time.monotonic() + 60)
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"Server exited during startup with status {process.returncode}")
                        try:
                            status, _, _ = client.request("/v1/internal/ping", timeout=2, raw=True)
                            if status == 200:
                                break
                        except (OSError, urllib.error.URLError):
                            pass
                        if time.monotonic() >= startup_deadline:
                            raise TimeoutError("Native server startup exceeded 60 seconds")
                        time.sleep(0.1)
                    report["ui"] = inspect_ui(client, manifest)
                    listing = client.expect(PREFIX + "checkpoints")
                    if not isinstance(listing, dict) or not isinstance(listing.get("models"), list):
                        raise RuntimeError("Native checkpoint listing is invalid")
                    if not any(item.get("name") == model.name for item in listing["models"] if isinstance(item, dict)):
                        raise RuntimeError("Pinned model is missing from the native checkpoint index")
                    baseline = client.expect(PREFIX + "options")
                    if baseline.get("sd_model_checkpoint") != "":
                        raise RuntimeError("Fresh native server unexpectedly has a selected checkpoint")
                    report["baseline_options"] = baseline
                    devices = client.expect(PREFIX + "backend-devices")
                    report["devices"] = devices
                    requested_backend = args.device if args.backend == "webgpu" and args.device != "auto" else args.backend
                    if args.backend == "webgpu":
                        selector = devices.get("default_webgpu_selector") if args.device == "auto" else args.device
                        selected = [item for item in devices.get("devices", [])
                                    if item.get("selector") == selector or item.get("stable_id") == selector]
                        if len(selected) != 1:
                            raise RuntimeError("Requested WebGPU device is not uniquely available")
                        report["selected_device"] = selected[0]
                        if args.require_hardware and (selected[0].get("software") is not False or
                                selected[0].get("provider") != "native" or
                                selected[0].get("type") not in ("gpu", "integrated-gpu")):
                            raise RuntimeError("Hardware inference required, but the selected adapter is not a native physical GPU")
                    report["generation"] = generation_checks(client, model.name, baseline, process, deadline, logs, requests,
                                                               args.steps, requested_backend)
                    report["invalid_overrides"] = invalid_override_checks(client, model.name, baseline)
                    if args.skip_cancel:
                        report["cancellation"] = {"status": "not-run", "reason": "explicit --skip-cancel"}
                    else:
                        report["cancellation"] = cancellation_check(client, model.name, baseline, process, deadline,
                                                                     requests, args.cancel_steps, requested_backend)
                    if process.poll() is not None:
                        raise RuntimeError("Server exited after inference instead of remaining available")
                    main_lane = platform.system() == "Linux" and args.provider != "embedded"
                    marker = "NATIVE_INFERENCE_MAIN" if main_lane else "NATIVE_INFERENCE_WORKER"
                    execution_log = stderr.read_text(errors="replace")
                    worker_records = re.findall(
                        r"^" + marker + r" stack_bytes=(\d+) guard_bytes=(\d+) "
                        r"stack_source=pthread_getattr_np$",
                        execution_log, re.MULTILINE)
                    expected_workers = 1 if args.skip_cancel else 2
                    if len(worker_records) != expected_workers or any(
                            (int(size) < 8 * 1024 * 1024 if main_lane else int(size) != 8 * 1024 * 1024)
                            or (not main_lane and int(guard) <= 0)
                            for size, guard in worker_records):
                        raise RuntimeError("Native requests did not report their actual inference execution stacks")
                    report["inference_workers"] = [
                        {"stack_bytes": int(size), "guard_bytes": int(guard),
                         "source": "pthread_getattr_np", "lane": "original-main" if main_lane else "worker"}
                        for size, guard in worker_records]
                    execution_records = re.findall(
                        r"^NATIVE_INFERENCE_EXECUTION success=([01]) selector=(\S+) provider=(\S+) "
                        r"software=(-?\d+) adapter_type=(\d+) graphs=(\d+) submissions=(\d+) "
                        r"dispatches=(\d+) matmuls=(\d+) readbacks=(\d+) native_loader_opens=(\d+)$",
                        execution_log, re.MULTILINE)
                    if len(execution_records) != expected_workers:
                        raise RuntimeError("Missing scoped native inference execution counters")
                    report["inference_execution"] = []
                    for index, record in enumerate(execution_records):
                        success, selector, provider, software, adapter_type, *values = record
                        counts = dict(zip(("graphs", "submissions", "dispatches", "matmuls", "readbacks", "native_loader_opens"), map(int, values)))
                        entry = {"success": success == "1", "selector": selector, "provider": provider,
                                 "software": int(software), "adapter_type": int(adapter_type), **counts}
                        report["inference_execution"].append(entry)
                        if index == 0 and success != "1":
                            raise RuntimeError("The successful image request has no successful execution record")
                        if args.backend == "webgpu" and index == 0:
                            selected = report["selected_device"]
                            if selector != selected["selector"] or provider != selected["provider"]:
                                raise RuntimeError("Inference execution used a different WebGPU device")
                            if bool(int(software)) != selected["software"]:
                                raise RuntimeError("Inference software classification differs from device metadata")
                            if any(counts[key] <= 0 for key in ("graphs", "submissions", "dispatches", "matmuls", "readbacks")):
                                raise RuntimeError("Image inference did not execute and read back WebGPU work")
                            if provider == "native" and counts["native_loader_opens"] <= 0:
                                raise RuntimeError("Native provider did not open the system Vulkan loader")
                            if args.require_hardware and (int(software) != 0 or provider != "native" or int(adapter_type) not in (1, 2)):
                                raise RuntimeError("Hardware-required execution was not on a native discrete or integrated GPU")
                        if args.backend == "cpu" and any(counts[key] for key in ("graphs", "submissions", "dispatches", "matmuls", "readbacks")):
                            raise RuntimeError("CPU-only request unexpectedly executed WebGPU work")
                    report["status"] = "passed"
                finally:
                    report["server_exit_before_cleanup"] = process.poll()
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                    report["server_cleanup_exit_status"] = process.returncode
                    for request in requests:
                        request.thread.join(timeout=2)
    except Exception as error:
        report["error"] = str(error)
        report["status"] = "failed"
        print(f"Native API verification failed: {error}", file=sys.stderr)
    finally:
        try:
            if "sha256_before" in report:
                report["sha256_after"] = sha256(app)
                report["app_hash_unchanged"] = report["sha256_after"] == report["sha256_before"]
            if "sha256_before" in report["model"]:
                report["model"]["sha256_after"] = sha256(model)
                report["model_hash_unchanged"] = report["model"]["sha256_after"] == report["model"]["sha256_before"]
        except Exception as error:
            report.update(status="failed", integrity_check_error=str(error))
        if report.get("app_hash_unchanged") is False or report.get("model_hash_unchanged") is False:
            report.update(status="failed", error="Application or checkpoint bytes changed during API verification")
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["evidence"] = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                              for path in (logs / "server.stdout.log", logs / "server.stderr.log", logs / "api-image.png")
                              if path.is_file()]
        (logs / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-pin", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--backend", choices=("cpu", "webgpu"), default="cpu")
    parser.add_argument("--provider", choices=("auto", "native", "embedded"), default="embedded")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--vulkan-icd", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--lavapipe-threads", type=int, default=2)
    parser.add_argument("--cancel-steps", type=int, default=8)
    parser.add_argument("--skip-cancel", action="store_true", help="Explicitly omit the second image request; report cancellation as not run")
    args = parser.parse_args()
    if args.require_hardware and (args.backend != "webgpu" or args.provider == "embedded"):
        parser.error("--require-hardware needs --backend webgpu and provider native or auto")
    if not 2 <= args.steps <= 100:
        parser.error("--steps must be in 2..100")
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 14400:
        parser.error("--timeout must be finite and in 1..14400 seconds")
    if not 1 <= args.lavapipe_threads <= 64 or not 2 <= args.cancel_steps <= 100:
        parser.error("Verification bounds: lavapipe threads 1..64 and cancel steps 2..100")
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
