#!/usr/bin/env python3
"""Exercise real native HTTP lifecycle with a deliberately substituted model call."""
import hashlib
import json
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from verify_inference_api import Client, PendingRequest, PREFIX, task_body, progress
from verify_runtime import runtime_environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active(client, pending, task_id, process, deadline):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Probe server exited while waiting for task activation")
        state = progress(client, task_id, timeout=0.5)
        if state is not None and not state["completed"]:
            return state
        if pending is not None and pending.done.is_set():
            raise RuntimeError("Probe request completed before active observation")
        time.sleep(0.01)
    raise TimeoutError("Probe task did not become active")


def main():
    app = Path(sys.argv[1]).resolve()
    logs = Path(sys.argv[2]).resolve()
    logs.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 45
    report = {"scope": "Real Crow routes/dispatcher with a two-second replacement of the model work",
              "real_model_called": False, "real_png_generated": False,
              "status": "failed", "http_events": [], "app_sha256_before": digest(app)}
    requests = []
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="work-", dir=logs) as temporary:
            work = Path(temporary)
            model = work / "synthetic-http-probe.gguf"
            model.write_bytes(b"diagnostic placeholder; model loader is not called\n")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [str(ROOT / "out/ape-x86_64.elf"), str(app), "sdkit", "--backend", "cpu",
                       "--ckpt-dir", str(work), "--port", str(port)]
            report["command"] = command
            environment = runtime_environment(work)
            environment["LP_NUM_THREADS"] = "2"
            client = Client(f"http://127.0.0.1:{port}", deadline, report["http_events"])
            with (logs / "server.stdout.log").open("wb") as stdout, (logs / "server.stderr.log").open("wb") as stderr:
                process = subprocess.Popen(command, cwd=work, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr)
                try:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError("Probe server exited during startup")
                        try:
                            if client.request("/v1/internal/ping", timeout=0.5, raw=True)[0] == 200:
                                break
                        except OSError:
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError("Probe startup timeout")
                        time.sleep(0.02)
                    listing = client.expect(PREFIX + "checkpoints")
                    if not any(item["name"] == model.name for item in listing["models"]):
                        raise RuntimeError("Synthetic model was not indexed")
                    baseline = client.expect(PREFIX + "options")
                    body = task_body(model.name, 4)
                    first = PendingRequest(client, body, 15)
                    requests.append(first)
                    active(client, first, body["force_task_id"], process, deadline)
                    latency = []
                    for _ in range(32):
                        tick = time.monotonic()
                        state = progress(client, body["force_task_id"], timeout=0.5)
                        latency.append(time.monotonic() - tick)
                        if state is None or state["completed"]:
                            raise RuntimeError("Synthetic work was not active for all progress probes")
                    client.expect(PREFIX + "txt2img", task_body(model.name, 4), status=409, timeout=0.5)
                    client.expect(PREFIX + "options", baseline, status=409, timeout=0.5)
                    status, response, _ = first.result(deadline)
                    if status != 200 or response.get("images") != ["cHJvYmU="]:
                        raise RuntimeError("Synthetic result did not return through the actual server handler")
                    if not progress(client, body["force_task_id"])["completed"]:
                        raise RuntimeError("Synthetic task was not completed")
                    report["progress"] = {"requests": 32, "max_seconds": max(latency),
                                          "all_seconds": latency, "concurrent_generation": 409,
                                          "concurrent_options": 409}
                    # Disconnect after a real asynchronous request is accepted.
                    disconnected = task_body(model.name, 4)
                    payload = json.dumps(disconnected).encode()
                    connection = socket.create_connection(("127.0.0.1", port), timeout=1)
                    connection.sendall((f"POST {PREFIX}txt2img HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                                        "Connection: close\r\n\r\n").encode() + payload)
                    active(client, None, disconnected["force_task_id"], process, deadline)
                    connection.close()
                    while True:
                        state = progress(client, disconnected["force_task_id"], timeout=0.5)
                        if state and state["completed"]:
                            break
                        time.sleep(0.05)
                    client.expect("/v1/internal/ping", timeout=0.5)
                    report["disconnect"] = {"task_completed": True, "server_responded_afterward": True}
                    # Stop while another asynchronous response is still pending.
                    stopping = task_body(model.name, 4)
                    last = PendingRequest(client, stopping, 15)
                    requests.append(last)
                    active(client, last, stopping["force_task_id"], process, deadline)
                    while "HTTP_PROBE_WORK_STARTED call=3 " not in (logs / "server.stderr.log").read_text():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Third synthetic workload did not begin")
                        time.sleep(0.01)
                    tick = time.monotonic()
                    process.send_signal(signal.SIGTERM)
                    code = process.wait(timeout=5)
                    if code != 0:
                        raise RuntimeError(f"Graceful probe shutdown returned {code}")
                    report["shutdown"] = {"signal": "SIGTERM", "exit_status": code,
                                          "seconds": time.monotonic() - tick,
                                          "request_active_at_signal": True}
                    report["status"] = "passed"
                finally:
                    report["exit_before_cleanup"] = process.poll()
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                    report["cleanup_exit"] = process.returncode
                    for request in requests:
                        request.thread.join(timeout=2)
    except Exception as error:
        report["error"] = str(error)
    report["app_sha256_after"] = digest(app)
    if report["app_sha256_before"] != report["app_sha256_after"]:
        report.update(status="failed", error="Probe executable changed during execution")
    report["elapsed_seconds"] = time.monotonic() - started
    (logs / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "http_events"}, indent=2))
    return report["status"] != "passed"


if __name__ == "__main__":
    raise SystemExit(main())
