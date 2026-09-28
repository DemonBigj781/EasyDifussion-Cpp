import os
import platform
import shutil
import subprocess
import requests
import hashlib
import concurrent.futures
import tarfile
import tempfile
import traceback

from easydiffusion.app import ROOT_DIR, getConfig
from easydiffusion.backend_args import parse_backend_commandline_args
from easydiffusion.utils import log

from common import run
import webui_common
from webui_common import (
    ping,
    load_model,
    unload_model,
    flush_model_changes,
    set_options,
    generate_images,
    generate_video,
    filter_images,
    get_url,
    stop_rendering,
    refresh_models,
    list_controlnet_filters,
    get_common_cli_args,
    create_context,
    do_start_backend,
    stop_backend,
)

ed_info = {
    "name": "sdkit3 backend for Easy Diffusion",
    "version": (1, 0, 0),
    "type": "backend",
}

BACKENDS_ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "backends"))
SDKIT3_BACKEND_DIR = os.path.join(BACKENDS_ROOT_DIR, "sdkit3")
os.makedirs(BACKENDS_ROOT_DIR, exist_ok=True)


def get_backend_dir():
    target = get_target()
    return os.path.join(SDKIT3_BACKEND_DIR, target)


BACKEND_BINARY_URL_BASE = "https://github.com/easydiffusion/sdkit/releases/download"
DEFAULT_BACKEND_VERSION = "v3.4.1"

OS_NAME = platform.system()


def install_backend():
    update_backend()


def update_backend():
    target = get_target()
    backend_dir = os.path.join(BACKENDS_ROOT_DIR, "sdkit3", target)

    config = getConfig()
    backend_config = config.get("backend_config") or {}

    if os.path.exists(backend_dir):
        print("Updating sdkit3 backend..")
    else:
        print("Installing sdkit3 backend..")

    print("Looking for backend build for target:", target)

    backend_version = backend_config.get("version", DEFAULT_BACKEND_VERSION)
    backend_binary_url = f"{BACKEND_BINARY_URL_BASE}/{backend_version}"
    manifest_url = f"{backend_binary_url}/{target}-manifest.json"

    print(f"Fetching manifest from {manifest_url}")
    try:
        response = requests.get(manifest_url)
    except:
        print("Wasn't able to fetch the manifest.")
        traceback.print_exc()
        return

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise ValueError(
                f"Target platform {target} does not exist. Please post a message on our Discord server ( https://discord.com/invite/u9yhsFmEkB ) or create a new issue at https://github.com/easydiffusion/sdkit/issues to request this platform build."
            )
        else:
            raise
    manifest = response.json()
    files = manifest["files"]

    os.makedirs(backend_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(update_or_download_file, filename, info, backend_binary_url, backend_dir)
            for filename, info in files.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("Backend update complete.")


def update_or_download_file(filename, info, base_url, backend_dir):
    from sdkit.utils import download_file

    filepath = os.path.join(backend_dir, filename)
    expected_sha = info["sha256"]
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        if actual_sha == expected_sha:
            print(f"File {filename} is up to date.")
            return

    # download
    uri = info["uri"]
    download_url = f"{base_url}/{uri}"
    print(f"Downloading {filename} from {download_url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        download_file(download_url, tmp_path)

        # extract
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(backend_dir)
        print(f"Extracted {filename}")
    finally:
        os.unlink(tmp_path)


def start_backend():
    config = getConfig()
    backend_config = config.get("backend_config") or {}

    backend_dir = get_backend_dir()
    log.info(f"Backend dir: {backend_dir}")

    was_still_installing = not is_installed()

    if backend_config.get("auto_update", True) or not is_installed():
        update_backend()

    user_args = backend_config.get("COMMANDLINE_ARGS")
    user_args = parse_backend_commandline_args(user_args or [])

    webui_common.WEBUI_API_PREFIX = "/v1"
    webui_common.USE_SDKIT3_API = True

    def run_fn():
        exe_name = "sdkit.exe" if OS_NAME == "Windows" else "sdkit"
        executable = os.path.join(backend_dir, exe_name)
        common_cli_args = get_common_cli_args(return_string=False)
        cmd = [executable] + common_cli_args + user_args

        binary_stat = os.stat(executable)
        log.info(
            f"starting: {cmd} (binary size={binary_stat.st_size}, "
            f"mtime_ns={binary_stat.st_mtime_ns})"
        )

        environment = get_backend_environment(backend_dir)
        return run(
            cmd,
            cwd=backend_dir,
            env=environment,
            wait=False,
            output_prefix="[sdkit3] ",
        )

    do_start_backend(was_still_installing, run_fn)


def uninstall_backend():
    shutil.rmtree(SDKIT3_BACKEND_DIR)


def is_installed():
    backend_dir = get_backend_dir()
    exe_name = "sdkit.exe" if OS_NAME == "Windows" else "sdkit"
    if os.path.exists(os.path.join(backend_dir, exe_name)):
        return True

    return False


def get_backend_devices():
    response = webui_common.webui_get("/sdapi/v1/backend-devices", timeout=5)
    response.raise_for_status()
    payload = response.json()
    devices = payload.get("devices", [])
    return devices if isinstance(devices, list) else []


def get_backend_environment(backend_dir):
    """Return the process environment required by a local native bundle."""
    environment = os.environ.copy()
    if "-sycl-" not in os.path.basename(backend_dir):
        return environment

    setvars_path = environment.get("ONEAPI_SETVARS", "/opt/intel/oneapi/setvars.sh")
    if not os.path.isfile(setvars_path):
        log.warning(
            "Intel oneAPI setvars.sh was not found at %s; the SYCL backend "
            "may be unable to load its runtime libraries.",
            setvars_path,
        )
        return environment

    try:
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1" --force >/dev/null && env -0',
                "easy-diffusion-oneapi",
                setvars_path,
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        for entry in completed.stdout.split(b"\0"):
            if not entry or b"=" not in entry:
                continue
            name, value = entry.split(b"=", 1)
            environment[os.fsdecode(name)] = os.fsdecode(value)
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning("Unable to load the Intel oneAPI runtime environment: %s", exc)

    return environment


def get_target():
    """Get the target for the build."""
    config = getConfig()
    backend_config = config.get("backend_config") or {}

    def get_os():
        """Get OS name for target."""
        if OS_NAME == "Windows":
            return "win"
        elif OS_NAME == "Darwin":
            return "mac"
        elif OS_NAME == "Linux":
            return "linux"
        else:
            return OS_NAME.lower()

    def get_arch():
        """Get architecture for target."""
        processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
        machine = platform.machine().lower()

        if processor_identifier.startswith("arm") and machine.endswith("64"):
            return "arm64"

        if machine in ("x86_64", "amd64"):
            return "x64"
        elif machine in ("arm64", "aarch64"):
            return "arm64"
        else:
            return machine

    platform_name = backend_config.get("platform")
    # Resolve an explicit "auto" value as well as a missing setting. Auto
    # always selects one runtime platform; CUDA/ROCm take priority over Vulkan.
    if platform_name == "auto-cuda":
        platform_name = "cuda"
    elif platform_name == "auto-vulkan":
        platform_name = "vulkan"
    elif not platform_name or platform_name == "auto":
        platform_name = get_platform_name()
    variant_name = backend_config.get("variant", get_variant_name(platform_name))

    target = f"{get_os()}-{get_arch()}-{platform_name}-{variant_name}"

    return target


def get_platform_name():
    if OS_NAME == "Darwin":
        return "metal"

    # use torchruntime to determine if nvidia gpu is present
    from torchruntime.device_db import get_gpus
    from torchruntime.platform_detection import get_torch_platform

    gpus = get_gpus()
    torch_platform = get_torch_platform(gpus)

    if torch_platform == "cpu":
        return "cpu"

    normalized_platform = str(torch_platform).lower()
    if normalized_platform.startswith("rocm") or normalized_platform.startswith("hip"):
        return "rocm"

    # If a GPU supports both CUDA and Vulkan, torchruntime reports CUDA here;
    # keep Automatic exclusive to that single backend.
    if normalized_platform.startswith("cu") or normalized_platform.startswith("cuda"):
        return "cuda"

    return "vulkan"


def get_variant_name(platform_name):
    if platform_name in ("cuda", "cuda-vulkan"):
        # deduce the variant from gpu compute capability
        from torchruntime.device_db import get_gpus
        from torchruntime.gpu_db import get_nvidia_arch
        from torchruntime.consts import NVIDIA

        gpus = get_gpus()
        for gpu in gpus:
            if gpu.vendor_id == NVIDIA and gpu.is_discrete:
                arch = get_nvidia_arch([gpu.device_name])  # 7.5, 12 etc
                arch = int(arch * 10)
                return f"sm{arch}"

    return "any"


# Sprite-GPT occupies the ordinary image model slot but runs in its own native
# process. Never send its manifest to stable-diffusion.cpp as a checkpoint.
def load_model(context, model_type, **kwargs):
    from easydiffusion import sprite_gpt

    model_path = context.model_paths.get(model_type)
    if model_type == "stable-diffusion":
        sprite_gpt.unload_model(context)
        if sprite_gpt.is_sprite_model(model_path):
            sprite_gpt.load_model(context, model_path, os.path.join(get_backend_dir(), "sdkit-sprite-gpt"))
            context.sprite_gpt_restart_required = bool(webui_common.curr_models[model_type])
            webui_common.curr_models[model_type] = None
            return
    return webui_common.load_model(context, model_type, **kwargs)


def unload_model(context, model_type, **kwargs):
    if model_type == "stable-diffusion":
        from easydiffusion import sprite_gpt
        sprite_gpt.unload_model(context)
    return webui_common.unload_model(context, model_type, **kwargs)


def flush_model_changes(context):
    if getattr(context, "sprite_gpt_model", None):
        # The native server applies model options lazily. Clear its selection
        # and restart once when switching from a retained diffusion checkpoint
        # so that checkpoint does not keep the GPU memory needed by Sprite-GPT.
        response = webui_common.webui_post("/sdapi/v1/options", json={
            "sd_model_checkpoint": None, "forge_additional_modules": [],
        })
        response.raise_for_status()
        if getattr(context, "sprite_gpt_restart_required", False):
            from easydiffusion.backend_manager import restart_backend
            restart_backend()
            context.sprite_gpt_restart_required = False
        return
    return webui_common.flush_model_changes(context)


def set_options(context, **kwargs):
    if getattr(context, "sprite_gpt_model", None):
        context.sprite_gpt_options = kwargs
        return
    return webui_common.set_options(context, **kwargs)


def generate_images(context, **kwargs):
    if getattr(context, "sprite_gpt_model", None):
        from easydiffusion import sprite_gpt
        return sprite_gpt.generate_images(context, **kwargs)
    return webui_common.generate_images(context, **kwargs)


def stop_rendering(context):
    if getattr(context, "sprite_gpt_model", None):
        from easydiffusion import sprite_gpt
        sprite_gpt.stop_rendering(context)
        return
    return webui_common.stop_rendering(context)
