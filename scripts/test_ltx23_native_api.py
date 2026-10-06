"""Validate the native video API against an isolated, empty model store."""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--runtime", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ltx23-api-") as directory:
        directory = pathlib.Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "LD_LIBRARY_PATH": str(args.runtime.resolve()), "OMP_NUM_THREADS": "1"}
        log = (directory / "server.log").open("w+")
        process = subprocess.Popen([str(args.binary.resolve()), "--port", str(port), "--backend", "cpu",
                                    "--ckpt-dir", str(directory), "--vae-dir", str(directory),
                                    "--text-encoder-dir", str(directory)], cwd=directory, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        origin = f"http://127.0.0.1:{port}/v1"
        try:
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Isolated native server exited before startup")
                try:
                    with urllib.request.urlopen(origin + "/internal/ping", timeout=1) as response:
                        assert response.status == 200
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Isolated native server did not become ready")
                    time.sleep(0.1)
            for route in ("txt2video", "img2video"):
                for field in ("audio_vae_path", "embeddings_connectors_path"):
                    for value, expected in (([], f"{field} must be a model path string"),
                                            (str(directory / "missing.safetensors"), "Video companion file not found")):
                        # A nonempty image field reaches resource validation, but no
                        # image decoding or generation can occur without a checkpoint.
                        payload = {field: value, "init_images": ["not-decoded"]}
                        request = urllib.request.Request(origin + "/sdapi/v1/" + route,
                            json.dumps(payload).encode(), {"Content-Type": "application/json"})
                        try:
                            urllib.request.urlopen(request, timeout=5)
                            raise AssertionError("Invalid video resources were accepted")
                        except urllib.error.HTTPError as error:
                            body = json.load(error)
                            assert error.code == 500, body
                            assert expected in body["message"], body
            with urllib.request.urlopen(origin + "/sdapi/v1/options", timeout=5) as response:
                options = json.load(response)
                assert not options.get("sd_model_checkpoint"), options
            print("PASS: native txt2video/img2video reject invalid and missing audio-VAE/connector paths; no checkpoint loaded")
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read()[-8000:])
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


if __name__ == "__main__":
    main()
