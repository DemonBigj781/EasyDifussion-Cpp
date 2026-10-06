#!/usr/bin/env python3
"""Generate a fixed-seed validation image with a finished SD1.5 LoRA."""
import argparse
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def decode_messages(text):
    decoder = json.JSONDecoder()
    messages = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        try:
            item, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            break
        messages.append(item)
        offset = end
    return messages


def request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True, help="Use the dataset caption; no placeholder is added")
    parser.add_argument("--url", default="http://127.0.0.1:10000")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_model = Path(args.base_model).expanduser().resolve()
    lora = Path(args.lora).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not base_model.is_file():
        raise SystemExit(f"Base checkpoint not found: {base_model}")
    if not lora.is_file():
        raise SystemExit(f"LoRA checkpoint not found: {lora}")

    payload = {
        "prompt": args.prompt,
        "negative_prompt": "",
        "seed": args.seed,
        "width": 512,
        "height": 512,
        "num_outputs": 1,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "sampler_name": "euler_a",
        "output_format": "png",
        "output_quality": 100,
        "use_stable_diffusion_model": str(base_model),
        "use_lora_model": [str(lora)],
        "lora_alpha": [args.weight],
        "session_id": "sd15-lora-validation",
        "temporary_output": True,
        "show_only_filtered_image": True,
        "stream_image_progress": False,
    }

    root = args.url.rstrip("/")
    try:
        start = json.loads(request_json(root + "/render", payload))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not start LoRA sample through {root}: {exc}") from exc
    task_id = start.get("task")
    if not task_id:
        raise SystemExit(f"Render API did not return a task id: {start}")
    stream_url = urllib.parse.urljoin(root + "/", start.get("stream", f"/image/stream/{task_id}"))
    print(f"Sampling LoRA {lora.name}: prompt={args.prompt!r}, seed={args.seed}, weight={args.weight}", flush=True)

    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        try:
            body = request_json(stream_url, timeout=120)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 425, 410):
                time.sleep(1)
                continue
            raise SystemExit(f"LoRA sample request failed: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except (OSError, urllib.error.URLError) as exc:
            time.sleep(1)
            continue

        for message in decode_messages(body):
            if not isinstance(message, dict):
                continue
            if message.get("status") == "failed":
                raise SystemExit(f"LoRA sample failed: {message.get('detail', message)}")
            if message.get("status") == "succeeded":
                outputs = message.get("output") or []
                if not outputs or not outputs[0].get("data"):
                    raise SystemExit(f"Sample completed without image data: {message}")
                image_data = outputs[0]["data"]
                if image_data.startswith("data:"):
                    image_data = image_data.split(",", 1)[1]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(base64.b64decode(image_data, validate=True))
                print(f"Saved LoRA validation image: {output.resolve()}", flush=True)
                return
        time.sleep(1)

    raise SystemExit("Timed out waiting for the LoRA sample render")


if __name__ == "__main__":
    main()
