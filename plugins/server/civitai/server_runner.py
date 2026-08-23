import os
import socket
import threading
import time
import uvicorn

from .app import app

HOST = os.environ.get("CIVITAI_SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("CIVITAI_SERVER_PORT", "9004"))

_server_started = False


def _port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _run_server():
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",
        access_log=False,
    )


def ensure_server_running():
    global _server_started

    if _server_started:
        return

    if _port_open(HOST, PORT):
        _server_started = True
        return

    thread = threading.Thread(
        target=_run_server,
        daemon=True,
        name="CivitAI-Server",
    )
    thread.start()

    for _ in range(20):
        if _port_open(HOST, PORT):
            _server_started = True
            return
        time.sleep(0.1)

    print("[CivitAI] Warning: server thread started but port not responding yet")


if __name__ == "__main__":
    ensure_server_running()
    while True:
        time.sleep(3600)
