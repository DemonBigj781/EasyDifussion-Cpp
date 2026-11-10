"""Disposable child processes only; never import or start Easy Diffusion."""
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time

mode, *args = sys.argv[1:]
if mode == "record":
    Path(args[0]).write_text(json.dumps({
        "args": args[2:], "cwd": os.getcwd(),
        "bundle": os.environ.get("EDCPP_LAUNCH_BUNDLE"),
        "value": os.environ.get("FIXTURE_VALUE"),
        "removed": os.environ.get("FIXTURE_REMOVED"),
    }), encoding="utf-8")
    raise SystemExit(int(args[1]))
if mode == "probe":
    Path(args[0]).write_text("probe ran", encoding="utf-8")
    time.sleep(float(args[2]))
    raise SystemExit(int(args[1]))
if mode in ("server", "transient-server"):
    server = socket.socket()
    server.bind(("127.0.0.1", int(args[0])))
    server.listen()
    if len(args) > 1:
        child = subprocess.Popen([sys.executable, __file__, "sleep"])
        Path(args[1]).write_text(str(child.pid), encoding="ascii")
    server.settimeout(.05)
    deadline = time.monotonic() + .4
    while mode != "transient-server" or time.monotonic() < deadline:
        try:
            client, _ = server.accept()
            client.close()
        except socket.timeout:
            pass
    raise SystemExit(9)
if mode == "exit":
    raise SystemExit(int(args[0]))
if mode in ("sleep", "stubborn"):
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if args:
        Path(args[0]).write_text(str(os.getpid()), encoding="ascii")
    while True:
        time.sleep(1)
