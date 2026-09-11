#!/usr/bin/env python3
"""Run a cold/warm sdkit image-generation benchmark on Jetson Xavier."""

import argparse
import atexit
import base64
import binascii
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
STANDARD_CLIP_CONTEXT_TOKENS = 77
STANDARD_CLIP_PAYLOAD_TOKENS = STANDARD_CLIP_CONTEXT_TOKENS - 2
PROMPT_TAIL_MARKER = "A puppy"
RELOAD_MARKERS = (
    "Model change detected, loading new model",
    "Initializing SD context with model",
    "loading diffusion model from",
)
CSV_FIELDS = (
    "run_id",
    "timestamp_utc",
    "status",
    "phase",
    "wall_ms",
    "valid",
    "png_bytes",
    "model_reload_detected",
    "request_steps_per_second",
    "sampling_seconds",
    "sampling_steps_per_second",
    "model_initialization_ms",
    "tensor_load_seconds",
    "prompt_context_tokens",
    "tail_beyond_standard_clip",
    "longclip_native_context_tokens",
)
LOG_TIMESTAMP = re.compile(r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
SAMPLING_SECONDS = re.compile(
    r"\bsampling completed, taking (?P<seconds>[0-9]+(?:\.[0-9]+)?)s\b"
)
TENSOR_LOAD_SECONDS = re.compile(
    r"\bloading tensors completed, taking (?P<seconds>[0-9]+(?:\.[0-9]+)?)s\b"
)
LONGCLIP_CONTEXT_TOKENS = re.compile(
    r"\bLongCLIP detected\b[^\r\n]*\(native context: (?P<tokens>[0-9]+) tokens\)"
)


class BenchmarkError(RuntimeError):
    pass


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def utc_text(value=None):
    value = value or utc_now()
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def command_output(arguments, timeout=10):
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    return completed.stdout.strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def model_metadata(path):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BenchmarkError(f"Model file not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def count_clip_payload_tokens(token_counter, prompt):
    try:
        completed = subprocess.run(
            [str(token_counter), prompt],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise BenchmarkError(f"CLIP token counter failed: {error}") from error
    output = completed.stdout.strip()
    if completed.returncode != 0 or not output.isdigit():
        raise BenchmarkError(
            f"CLIP token counter exited with {completed.returncode}: {output or '<no output>'}"
        )
    return int(output)


def validate_prompt_token_budget(prompt, tail_marker, token_counter):
    tail_offset = prompt.find(tail_marker)
    if tail_offset < 0:
        raise BenchmarkError(f"Prompt does not contain semantic tail marker: {tail_marker}")
    payload_tokens = token_counter(prompt)
    prefix_tokens = token_counter(prompt[:tail_offset])
    context_tokens = payload_tokens + 2
    tail_beyond_standard_clip = prefix_tokens >= STANDARD_CLIP_PAYLOAD_TOKENS
    if context_tokens <= STANDARD_CLIP_CONTEXT_TOKENS:
        raise BenchmarkError(
            f"Prompt uses {context_tokens} CLIP context tokens; it must exceed the 77-token standard context"
        )
    if not tail_beyond_standard_clip:
        raise BenchmarkError(
            f"Semantic tail starts after {prefix_tokens} payload tokens; it must start beyond the standard "
            f"{STANDARD_CLIP_PAYLOAD_TOKENS}-token payload"
        )
    return {
        "tokenizer": "sdkit CLIP BPE",
        "payload_tokens": payload_tokens,
        "context_tokens": context_tokens,
        "standard_context_tokens": STANDARD_CLIP_CONTEXT_TOKENS,
        "tail_marker": tail_marker,
        "payload_tokens_before_tail": prefix_tokens,
        "tail_beyond_standard_clip": tail_beyond_standard_clip,
    }


def _paeth(left, above, upper_left):
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def validate_png(png_data, expected_width, expected_height):
    if not png_data.startswith(PNG_SIGNATURE):
        raise BenchmarkError("Response image is not a PNG")

    offset = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while offset + 12 <= len(png_data):
        length = struct.unpack(">I", png_data[offset : offset + 4])[0]
        chunk_end = offset + 12 + length
        if chunk_end > len(png_data):
            raise BenchmarkError("PNG contains a truncated chunk")
        chunk_type = png_data[offset + 4 : offset + 8]
        chunk_data = png_data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", png_data[offset + 8 + length : chunk_end])[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise BenchmarkError("PNG chunk CRC validation failed")
        if chunk_type == b"IHDR":
            if length != 13:
                raise BenchmarkError("PNG has an invalid IHDR chunk")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if compression != 0 or filtering != 0:
                raise BenchmarkError("PNG uses unsupported compression or filtering")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
        offset = chunk_end

    if width != expected_width or height != expected_height:
        raise BenchmarkError(
            f"PNG dimensions are {width}x{height}, expected {expected_width}x{expected_height}"
        )
    if bit_depth != 8 or color_type not in (0, 2, 4, 6) or interlace != 0:
        raise BenchmarkError(
            f"PNG format is unsupported for validation: bit_depth={bit_depth}, "
            f"color_type={color_type}, interlace={interlace}"
        )

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    bytes_per_pixel = channels
    stride = width * channels
    try:
        filtered = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise BenchmarkError(f"PNG decompression failed: {error}") from error
    expected_size = height * (stride + 1)
    if len(filtered) != expected_size:
        raise BenchmarkError(f"PNG pixel stream has {len(filtered)} bytes, expected {expected_size}")

    pixels = bytearray(height * stride)
    source_offset = 0
    for row_index in range(height):
        filter_type = filtered[source_offset]
        source_offset += 1
        row = filtered[source_offset : source_offset + stride]
        source_offset += stride
        destination_offset = row_index * stride
        for column, value in enumerate(row):
            left = pixels[destination_offset + column - bytes_per_pixel] if column >= bytes_per_pixel else 0
            above = pixels[destination_offset + column - stride] if row_index > 0 else 0
            upper_left = (
                pixels[destination_offset + column - stride - bytes_per_pixel]
                if row_index > 0 and column >= bytes_per_pixel
                else 0
            )
            if filter_type == 0:
                reconstructed = value
            elif filter_type == 1:
                reconstructed = value + left
            elif filter_type == 2:
                reconstructed = value + above
            elif filter_type == 3:
                reconstructed = value + ((left + above) // 2)
            elif filter_type == 4:
                reconstructed = value + _paeth(left, above, upper_left)
            else:
                raise BenchmarkError(f"PNG uses unknown row filter {filter_type}")
            pixels[destination_offset + column] = reconstructed & 0xFF

    unique_values = len(set(pixels))
    mean = sum(pixels) / len(pixels)
    variance = sum((value - mean) ** 2 for value in pixels) / len(pixels)
    if unique_values < 2 or not math.isfinite(variance) or variance <= 0:
        raise BenchmarkError("PNG is blank or has no measurable pixel variation")

    return {
        "valid": True,
        "format": "png",
        "width": width,
        "height": height,
        "channels": channels,
        "png_bytes": len(png_data),
        "unique_channel_values": unique_values,
        "pixel_byte_variance": round(variance, 3),
    }


def extract_and_validate_image(response_body, expected_width, expected_height):
    try:
        payload = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"txt2img response is not valid JSON: {error}") from error
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list) or not images or not isinstance(images[0], str):
        raise BenchmarkError("txt2img response does not contain an image")
    encoded = images[0]
    if encoded.startswith("data:"):
        try:
            encoded = encoded.split(",", 1)[1]
        except IndexError as error:
            raise BenchmarkError("txt2img response has an invalid data URI") from error
    try:
        png_data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise BenchmarkError(f"txt2img response contains invalid Base64: {error}") from error
    return validate_png(png_data, expected_width, expected_height)


def find_model_reload_markers(log_text):
    return [marker for marker in RELOAD_MARKERS if marker in log_text]


def extract_performance_metrics(log_text, steps, wall_ms):
    model_start = None
    model_end = None
    for line in log_text.splitlines():
        timestamp_match = LOG_TIMESTAMP.match(line)
        if timestamp_match is None:
            continue
        timestamp = dt.datetime.strptime(timestamp_match.group("timestamp"), "%Y-%m-%d %H:%M:%S.%f")
        if model_start is None and RELOAD_MARKERS[0] in line:
            model_start = timestamp
        elif model_start is not None and "SD context initialized successfully" in line:
            model_end = timestamp
            break

    sampling_matches = list(SAMPLING_SECONDS.finditer(log_text))
    sampling_seconds = float(sampling_matches[-1].group("seconds")) if sampling_matches else None
    tensor_load_seconds = round(
        sum(float(match.group("seconds")) for match in TENSOR_LOAD_SECONDS.finditer(log_text)),
        3,
    )
    longclip_match = LONGCLIP_CONTEXT_TOKENS.search(log_text)
    model_initialization_ms = None
    if model_start is not None and model_end is not None:
        model_initialization_ms = round((model_end - model_start).total_seconds() * 1000)

    return {
        "request_steps_per_second": round(steps / (wall_ms / 1000.0), 3),
        "sampling_seconds": sampling_seconds,
        "sampling_steps_per_second": round(steps / sampling_seconds, 3) if sampling_seconds else None,
        "model_initialization_ms": model_initialization_ms,
        "tensor_load_seconds": tensor_load_seconds,
        "longclip_native_context_tokens": int(longclip_match.group("tokens")) if longclip_match else None,
    }


def _assert_no_image_payloads(value, path="record"):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"image", "images", "base64", "response_body"}:
                raise BenchmarkError(f"History record contains an image payload at {path}.{key}")
            _assert_no_image_payloads(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_image_payloads(child, f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith("data:image/"):
        raise BenchmarkError(f"History record contains an image payload at {path}")


def _history_rows(record):
    for phase in ("startup", "cold", "warm"):
        result = record.get("results", {}).get(phase, {})
        yield {
            "run_id": record.get("run_id", ""),
            "timestamp_utc": record.get("timestamp_utc", ""),
            "status": record.get("status", ""),
            "phase": phase,
            "wall_ms": record.get("timings_ms", {}).get(phase, ""),
            "valid": result.get("valid", ""),
            "png_bytes": result.get("png_bytes", ""),
            "model_reload_detected": result.get("model_reload_detected", ""),
            "request_steps_per_second": result.get("request_steps_per_second", ""),
            "sampling_seconds": result.get("sampling_seconds", ""),
            "sampling_steps_per_second": result.get("sampling_steps_per_second", ""),
            "model_initialization_ms": result.get("model_initialization_ms", ""),
            "tensor_load_seconds": result.get("tensor_load_seconds", ""),
            "prompt_context_tokens": record.get("prompt_tokenization", {}).get("context_tokens", ""),
            "tail_beyond_standard_clip": record.get("prompt_tokenization", {}).get(
                "tail_beyond_standard_clip", ""
            ),
            "longclip_native_context_tokens": result.get("longclip_native_context_tokens", ""),
        }


def _rebuild_csv_history_unlocked(jsonl_path, csv_path):
    records = []
    if jsonl_path.exists():
        for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BenchmarkError(f"Invalid history JSONL at line {line_number}: {error}") from error
            _assert_no_image_payloads(record)
            records.append(record)

    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerows(_history_rows(record))
    temporary.replace(csv_path)


def rebuild_csv_history(jsonl_path, csv_path):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jsonl_path.parent / ".history.lock"
    with lock_path.open("a", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        _rebuild_csv_history_unlocked(jsonl_path, csv_path)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def append_history(jsonl_path, csv_path, record):
    _assert_no_image_payloads(record)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jsonl_path.parent / ".history.lock"
    with lock_path.open("a", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        with jsonl_path.open("a", encoding="utf-8") as jsonl_stream:
            json.dump(record, jsonl_stream, sort_keys=True, separators=(",", ":"))
            jsonl_stream.write("\n")
        _rebuild_csv_history_unlocked(jsonl_path, csv_path)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def read_meminfo():
    values = {}
    with pathlib.Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
        for line in stream:
            key, raw_value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "SwapTotal", "SwapFree"}:
                values[key] = int(raw_value.strip().split()[0]) * 1024
    return values


def read_process_status(pid):
    values = {}
    status_path = pathlib.Path(f"/proc/{pid}/status")
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator or key not in {"VmRSS", "VmHWM", "VmSize", "VmSwap", "Threads"}:
            continue
        fields = raw_value.strip().split()
        value = int(fields[0])
        values[key] = value * 1024 if len(fields) > 1 and fields[1] == "kB" else value
    return values


class ResourceSampler:
    def __init__(self, path, pid, interval_seconds):
        self.path = path
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="sdkit-resource-sampler", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, self.interval_seconds * 4))

    def _run(self):
        with self.path.open("a", encoding="utf-8") as stream:
            while not self.stop_event.is_set():
                sample = {
                    "timestamp_utc": utc_text(),
                    "monotonic_ms": time.monotonic_ns() // 1_000_000,
                    "load_average": list(os.getloadavg()),
                    "memory_bytes": read_meminfo(),
                    "process": read_process_status(self.pid),
                }
                stream.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                self.stop_event.wait(self.interval_seconds)


def port_is_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def http_request(url, method="GET", payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        body = error.read()
        raise BenchmarkError(
            f"HTTP {error.code} from {url} ({len(body)} response bytes): {error.reason}"
        ) from error
    except urllib.error.URLError as error:
        raise BenchmarkError(f"Request to {url} failed: {error.reason}") from error


def wait_for_ping(base_url, server_process, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if server_process.poll() is not None:
            raise BenchmarkError(f"sdkit exited during startup with code {server_process.returncode}")
        try:
            status, body = http_request(f"{base_url}/v1/internal/ping", timeout=2)
            if status == 200 and body.strip() == b"OK":
                return
            last_error = f"unexpected ping response: HTTP {status}, {body[:80]!r}"
        except BenchmarkError as error:
            last_error = str(error)
        time.sleep(0.2)
    raise BenchmarkError(f"sdkit startup timed out after {timeout_seconds}s: {last_error}")


def stop_process(process, timeout=10):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=timeout)


def read_log_since(path, offset):
    time.sleep(0.1)
    with path.open("rb") as stream:
        stream.seek(offset)
        return stream.read().decode("utf-8", errors="replace")


def run_generation(base_url, payload, server_log_path, timeout_seconds):
    log_offset = server_log_path.stat().st_size
    start_ns = time.perf_counter_ns()
    status, response_body = http_request(
        f"{base_url}/v1/sdapi/v1/txt2img",
        method="POST",
        payload=payload,
        timeout=timeout_seconds,
    )
    wall_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
    if status != 200:
        raise BenchmarkError(f"txt2img returned HTTP {status}")
    validation = extract_and_validate_image(response_body, payload["width"], payload["height"])
    validation["http_code"] = status
    validation["response_bytes"] = len(response_body)
    validation["wall_ms"] = wall_ms
    log_segment = read_log_since(server_log_path, log_offset)
    markers = find_model_reload_markers(log_segment)
    validation["model_reload_detected"] = bool(markers)
    validation["model_reload_markers"] = markers
    validation.update(extract_performance_metrics(log_segment, payload["steps"], wall_ms))
    return validation


def gather_environment(repo_root, server_binary):
    device_model_path = pathlib.Path("/proc/device-tree/model")
    device_model = "unavailable"
    if device_model_path.exists():
        device_model = device_model_path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
    l4t_path = pathlib.Path("/etc/nv_tegra_release")
    return {
        "device_model": device_model,
        "uname": command_output(["uname", "-a"]),
        "l4t_release": l4t_path.read_text(encoding="utf-8").strip() if l4t_path.exists() else "unavailable",
        "cuda": command_output(["nvcc", "--version"]),
        "nvpmodel": command_output(["nvpmodel", "-q"]),
        "jetson_clocks": command_output(["jetson_clocks", "--show"]),
        "python": sys.version,
        "server_binary": {
            "path": str(server_binary),
            "size_bytes": server_binary.stat().st_size,
            "sha256": sha256_file(server_binary),
        },
        "git": {
            "commit": command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
            "branch": command_output(["git", "-C", str(repo_root), "branch", "--show-current"]),
            "dirty": bool(command_output(["git", "-C", str(repo_root), "status", "--porcelain"])),
        },
    }


def write_summary(path, record):
    lines = [
        f"run_id: {record['run_id']}",
        f"timestamp_utc: {record['timestamp_utc']}",
        f"status: {record['status']}",
    ]
    for phase in ("startup", "cold", "warm"):
        timing = record.get("timings_ms", {}).get(phase)
        result = record.get("results", {}).get(phase, {})
        lines.append(
            f"{phase}: wall_ms={timing} valid={result.get('valid', '')} "
            f"png_bytes={result.get('png_bytes', '')} "
            f"model_reload_detected={result.get('model_reload_detected', '')} "
            f"request_steps_per_second={result.get('request_steps_per_second', '')} "
            f"sampling_steps_per_second={result.get('sampling_steps_per_second', '')} "
            f"model_initialization_ms={result.get('model_initialization_ms', '')} "
            f"tensor_load_seconds={result.get('tensor_load_seconds', '')} "
            f"longclip_native_context_tokens={result.get('longclip_native_context_tokens', '')}"
        )
    if record.get("error"):
        lines.append(f"error: {record['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(arguments):
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    output_root = arguments.output_dir.expanduser().resolve()
    if output_root == pathlib.Path("/"):
        raise BenchmarkError("Refusing to use the filesystem root as the output directory")
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    server_log_path = run_dir / "server.log"
    tegrastats_log_path = run_dir / "tegrastats.log"
    resource_log_path = run_dir / "resource-samples.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.txt"

    server_binary = arguments.server_binary.expanduser().resolve()
    if not server_binary.is_file() or not os.access(server_binary, os.X_OK):
        raise BenchmarkError(f"sdkit binary is missing or not executable: {server_binary}")
    if not port_is_available(arguments.port):
        raise BenchmarkError(f"Benchmark port {arguments.port} is already in use")
    token_counter = server_binary.parent / "clip-token-count"
    if not token_counter.is_file() or not os.access(token_counter, os.X_OK):
        raise BenchmarkError(
            f"CLIP token counter is missing: {token_counter}. Configure with -DSDKIT_BUILD_TESTS=ON "
            "and build the clip-token-count target."
        )

    models = {
        "unet": model_metadata(arguments.unet),
        "clip_l": model_metadata(arguments.clip),
        "vae": model_metadata(arguments.vae),
    }
    server_command = [
        str(server_binary),
        "--port",
        str(arguments.port),
        "--log-level",
        "info",
        "--ckpt-dir",
        str(arguments.unet.expanduser().resolve().parent),
        "--vae-dir",
        str(arguments.vae.expanduser().resolve().parent),
        "--text-encoder-dir",
        str(arguments.clip.expanduser().resolve().parent),
        "--attention",
        "split",
        "--vae-tiling",
        "--vae-tiles",
        "32",
        "--vae-tiled-overlap",
        "8",
    ]
    options_payload = {
        "CLIP_stop_at_last_layers": -1,
        "forge_additional_modules": [models["clip_l"]["path"], models["vae"]["path"]],
        "live_previews_enable": False,
        "samples_format": "png",
        "sd_model_checkpoint": pathlib.Path(models["unet"]["path"]).name,
    }
    generation_payload = {
        "prompt": arguments.prompt,
        "negative_prompt": arguments.negative_prompt,
        "width": arguments.width,
        "height": arguments.height,
        "steps": arguments.steps,
        "cfg_scale": arguments.cfg_scale,
        "seed": arguments.seed,
        "batch_size": 1,
        "sampler_name": arguments.sampler,
    }
    prompt_tokenization = validate_prompt_token_budget(
        arguments.prompt,
        PROMPT_TAIL_MARKER,
        lambda prompt: count_clip_payload_tokens(token_counter, prompt),
    )
    record = {
        "run_id": run_id,
        "timestamp_utc": utc_text(started_at),
        "status": "running",
        "arguments": {
            "command_line": sys.argv,
            "server_command": server_command,
            "options_request": options_payload,
            "generation_request": generation_payload,
            "request_timeout_seconds": arguments.request_timeout,
            "startup_timeout_seconds": arguments.startup_timeout,
            "telemetry_interval_ms": arguments.telemetry_interval_ms,
        },
        "models": models,
        "prompt_tokenization": prompt_tokenization,
        "environment": gather_environment(repo_root, server_binary),
        "timings_ms": {},
        "results": {},
        "artifacts": {
            "run_directory": str(run_dir),
            "server_log": str(server_log_path),
            "tegrastats_log": str(tegrastats_log_path),
            "resource_samples": str(resource_log_path),
            "summary": str(summary_path),
        },
    }

    server_process = None
    telemetry_process = None
    resource_sampler = None
    server_log_stream = None
    failure = None

    def cleanup():
        nonlocal server_process, telemetry_process, resource_sampler, server_log_stream
        if resource_sampler is not None:
            resource_sampler.stop()
            resource_sampler = None
        stop_process(telemetry_process)
        telemetry_process = None
        stop_process(server_process)
        server_process = None
        if server_log_stream is not None:
            server_log_stream.close()
            server_log_stream = None

    atexit.register(cleanup)
    try:
        server_log_stream = server_log_path.open("wb")
        startup_start_ns = time.perf_counter_ns()
        server_process = subprocess.Popen(
            server_command,
            stdout=server_log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        resource_sampler = ResourceSampler(
            resource_log_path,
            server_process.pid,
            arguments.telemetry_interval_ms / 1000.0,
        )
        resource_sampler.start()

        tegrastats = shutil.which("tegrastats")
        if tegrastats:
            telemetry_stream = tegrastats_log_path.open("wb")
            telemetry_process = subprocess.Popen(
                [tegrastats, "--interval", str(arguments.telemetry_interval_ms)],
                stdout=telemetry_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            telemetry_stream.close()
        else:
            tegrastats_log_path.write_text("tegrastats unavailable\n", encoding="utf-8")

        base_url = f"http://127.0.0.1:{arguments.port}"
        wait_for_ping(base_url, server_process, arguments.startup_timeout)
        record["timings_ms"]["startup"] = (time.perf_counter_ns() - startup_start_ns) // 1_000_000
        record["results"]["startup"] = {"valid": True, "server_pid": server_process.pid}

        options_status, options_body = http_request(
            f"{base_url}/v1/sdapi/v1/options",
            method="POST",
            payload=options_payload,
            timeout=arguments.request_timeout,
        )
        record["results"]["options"] = {
            "http_code": options_status,
            "response_bytes": len(options_body),
        }

        cold_payload = dict(generation_payload, force_task_id=f"{run_id}-cold")
        cold_result = run_generation(base_url, cold_payload, server_log_path, arguments.request_timeout)
        record["timings_ms"]["cold"] = cold_result["wall_ms"]
        record["results"]["cold"] = cold_result
        if not cold_result["model_reload_detected"]:
            raise BenchmarkError("Cold generation did not record a model-load marker")
        longclip_context = cold_result["longclip_native_context_tokens"]
        if longclip_context is None:
            raise BenchmarkError("Cold generation did not record LongCLIP native-context evidence")
        if longclip_context < prompt_tokenization["context_tokens"]:
            raise BenchmarkError(
                f"LongCLIP native context is {longclip_context} tokens, below the prompt's "
                f"{prompt_tokenization['context_tokens']} context tokens"
            )

        warm_payload = dict(generation_payload, force_task_id=f"{run_id}-warm")
        warm_result = run_generation(base_url, warm_payload, server_log_path, arguments.request_timeout)
        record["timings_ms"]["warm"] = warm_result["wall_ms"]
        record["results"]["warm"] = warm_result
        if warm_result["model_reload_detected"]:
            raise BenchmarkError(
                "Warm generation reloaded the model: " + ", ".join(warm_result["model_reload_markers"])
            )
        record["status"] = "passed"
    except BaseException as error:
        failure = error
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        cleanup()
        atexit.unregister(cleanup)
        record["finished_at_utc"] = utc_text()
        record["telemetry"] = {
            "tegrastats_samples": sum(1 for line in tegrastats_log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
            if tegrastats_log_path.exists()
            else 0,
            "resource_samples": sum(1 for line in resource_log_path.read_text(encoding="utf-8").splitlines() if line.strip())
            if resource_log_path.exists()
            else 0,
        }
        residuals = [
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file() and (path.suffix.lower() == ".png" or "base64" in path.name.lower())
        ]
        record["residual_image_artifacts"] = residuals
        if residuals and record["status"] == "passed":
            record["status"] = "failed"
            record["error"] = "Residual image-bearing artifacts: " + ", ".join(residuals)
            failure = BenchmarkError(record["error"])
        atomic_write_json(manifest_path, record)
        write_summary(summary_path, record)
        append_history(output_root / "history.jsonl", output_root / "history.csv", record)

    if failure is not None:
        raise failure
    return record, manifest_path


def parse_arguments(argv=None):
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    build_bin = repo_root / "source" / "sdkit3-port-source" / "build" / "local-linux-aarch64-jetpack5-cuda-sm72" / "bin" / "sdkit"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=pathlib.Path, default=build_bin)
    parser.add_argument(
        "--unet",
        type=pathlib.Path,
        default=pathlib.Path("/home/jack/projects/stable-diffusion-assets/extracted/sd15_unet-Q4_K_M.gguf"),
    )
    parser.add_argument(
        "--clip",
        type=pathlib.Path,
        default=pathlib.Path("/home/jack/projects/stable-diffusion-assets/longclip-L-text.safetensors"),
    )
    parser.add_argument(
        "--vae",
        type=pathlib.Path,
        default=pathlib.Path("/home/jack/projects/stable-diffusion-assets/extracted/furception_vae_1-0.safetensors"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("SDKIT_BENCHMARK_DIR", repo_root / "benchmark-results" / "xavier-sd1")),
    )
    parser.add_argument("--port", type=int, default=18189)
    parser.add_argument("--startup-timeout", type=int, default=60)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--telemetry-interval-ms", type=int, default=500)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--sampler", default="Euler a")
    parser.add_argument(
        "--prompt",
        default="A blissful painting by Vincent_van_gogh BREAK masterpiece, best_quality, highly_detailed, scenic_landscape, rolling_green_hills, blue_sky, soft_clouds, warm_light, vivid_colors, painterly_brushstrokes, textured_canvas, peaceful_mood, pastoral_scenery, spring_grass, wildflowers, distant_trees, cinematic_composition, atmospheric_perspective, tranquil_countryside, gentle_breeze, storybook_illustration, expressive_color, harmonious_palette, A puppy in the meadows AND under a sunset, calming /red and orange sky/, (a small creek to the left), wearing a dandelion:1.2 on the left ear, Resting in a field of daisies +++",
    )
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in ("startup_timeout", "request_timeout", "telemetry_interval_ms", "width", "height", "steps"):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return arguments


def main(argv=None):
    arguments = parse_arguments(argv)
    try:
        record, manifest_path = run_benchmark(arguments)
    except BenchmarkError as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("benchmark interrupted", file=sys.stderr)
        return 130
    print(
        f"benchmark passed: startup={record['timings_ms']['startup']} ms, "
        f"cold={record['timings_ms']['cold']} ms, warm={record['timings_ms']['warm']} ms"
    )
    cold = record["results"]["cold"]
    warm = record["results"]["warm"]
    print(
        f"throughput: cold={cold['sampling_steps_per_second']} sampling steps/s "
        f"({cold['request_steps_per_second']} end-to-end), "
        f"warm={warm['sampling_steps_per_second']} sampling steps/s "
        f"({warm['request_steps_per_second']} end-to-end)"
    )
    print(
        f"model load: initialization={cold['model_initialization_ms']} ms, "
        f"cold tensor loads={cold['tensor_load_seconds']} s"
    )
    print(
        f"prompt tokens: {record['prompt_tokenization']['context_tokens']} context tokens, "
        f"tail after {record['prompt_tokenization']['payload_tokens_before_tail']} payload tokens, "
        f"LongCLIP native context={cold['longclip_native_context_tokens']} tokens"
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
