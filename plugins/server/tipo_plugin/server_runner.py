import os
import socket
import threading
import time
import uvicorn

from .app import app, get_logger

HOST = os.environ.get("TIPO_SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("TIPO_SERVER_PORT", "9003"))

_server_started = False
_logger = get_logger()


def _port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _run_server():
    _logger.info("Starting TIPO server on %s:%s", HOST, PORT)
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
        _logger.info("TIPO server already running on %s:%s", HOST, PORT)
        return

    thread = threading.Thread(
        target=_run_server,
        daemon=True,
        name="TIPO-Server",
    )
    thread.start()
    _logger.info("TIPO server thread started")

    for _ in range(20):
        if _port_open(HOST, PORT):
            _server_started = True
            _logger.info("TIPO server is responding on %s:%s", HOST, PORT)
            return
        time.sleep(0.1)

    _logger.warning("TIPO server thread started but port not responding yet")


if __name__ == "__main__":
    ensure_server_running()
    while True:
        time.sleep(3600)
