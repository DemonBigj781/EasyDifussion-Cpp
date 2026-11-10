"""Per-render-thread backend contexts and their active-device registry."""

import threading


_thread_context = threading.local()
_contexts_lock = threading.RLock()
_contexts_by_device = {}


def bind_context(context):
    """Bind an initialized context only to the calling render thread."""

    _thread_context.value = context
    device = getattr(context, "device", None)
    if device is not None:
        with _contexts_lock:
            _contexts_by_device[device] = context
    return context


def current_context():
    context = getattr(_thread_context, "value", None)
    if context is None:
        raise RuntimeError("No Easy Diffusion runtime context is bound to this thread.")
    return context


def get_contexts():
    """Return a stable snapshot for control-plane operations such as Stop All."""

    with _contexts_lock:
        return list(_contexts_by_device.values())


def get_context(device):
    with _contexts_lock:
        return _contexts_by_device.get(device)


def release_context(context=None):
    context = context or getattr(_thread_context, "value", None)
    if context is None:
        return
    device = getattr(context, "device", None)
    with _contexts_lock:
        if _contexts_by_device.get(device) is context:
            del _contexts_by_device[device]
    if getattr(_thread_context, "value", None) is context:
        del _thread_context.value


def __getattr__(name):
    if name == "context":
        return current_context()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def init(device):
    """
    Initializes the fields that will be bound to this runtime's context, and sets the current torch device
    """

    from easydiffusion import device_manager
    from easydiffusion.backend_manager import backend

    context = backend.create_context()

    context.stop_processing = False
    context.temp_images = {}
    context.partial_x_samples = None
    context.model_load_errors = {}
    context.enable_codeformer = True

    device_manager.device_init(context, device)
    return bind_context(context)


def set_vram_optimizations(context):
    from easydiffusion.app import getConfig

    config = getConfig()
    vram_usage_level = config.get("vram_usage_level", "balanced")

    if hasattr(context, "vram_usage_level") and vram_usage_level != context.vram_usage_level:
        context.vram_usage_level = vram_usage_level
        return True

    return False
