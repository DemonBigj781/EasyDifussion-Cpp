import os
import sys
import importlib.util
import threading
import time
import traceback

from easydiffusion.utils import log

backend = None
curr_backend_name = None
_backend_restart_lock = threading.Lock()
NATIVE_BACKEND_NAME = "sdkit3"


def load_backend_module(module_path):
    mod_dir = os.path.dirname(module_path)
    sys.path.insert(0, mod_dir)
    spec = importlib.util.spec_from_file_location(NATIVE_BACKEND_NAME, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if mod_dir in sys.path:
        sys.path.remove(mod_dir)

    log.info(f"Loaded backend: {module}")

    return module


def start_backend():
    global backend, curr_backend_name

    module_path = os.path.join(os.path.dirname(__file__), "backends", f"{NATIVE_BACKEND_NAME}.py")
    if not os.path.isfile(module_path):
        raise RuntimeError(f"The native backend module is missing: {module_path}")

    if backend is not None:
        try:
            backend.stop_backend()
        except Exception:
            log.exception(traceback.format_exc())

    log.info(f"Loading backend: {NATIVE_BACKEND_NAME}")
    backend = load_backend_module(module_path)
    curr_backend_name = NATIVE_BACKEND_NAME

    try:
        backend.start_backend()
    except Exception:
        log.exception(traceback.format_exc())


def restart_backend(timeout=120):
    """Restart the active backend and wait until it accepts API requests."""
    global backend

    if backend is None:
        raise RuntimeError("No backend is loaded.")

    from easydiffusion import task_manager

    with _backend_restart_lock:
        task_manager.current_state = task_manager.ServerStates.LoadingModel
        backend.stop_backend()
        backend.start_backend()

        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                if backend.ping(timeout=1):
                    task_manager.current_state = task_manager.ServerStates.Online
                    task_manager.current_state_error = None
                    return
            except Exception as error:
                last_error = error
            time.sleep(0.25)

        task_manager.current_state = task_manager.ServerStates.Unavailable
        message = "Backend did not become ready after its settings reload."
        if last_error:
            message += f" Last error: {last_error}"
        task_manager.current_state_error = RuntimeError(message)
        raise task_manager.current_state_error
