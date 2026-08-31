"""
This script checks and installs the Python 3.13 runtime modules.

This script runs from the Easy Diffusion installation root.

"""

import os, sys
from importlib.metadata import version as pkg_version
import platform
import shutil
import subprocess
from pathlib import Path
from pprint import pprint
import threading

os_name = platform.system()
if os_name != "Linux":
    raise SystemExit("This local Easy Diffusion fork currently supports Linux only.")
if sys.version_info[:2] != (3, 13):
    raise SystemExit(f"Easy Diffusion requires Python 3.13; found {platform.python_version()}.")

modules_to_check = {
    "setuptools": "69.5.1",
    "numpy": "2.2.6",
    "scipy": "1.16.3",
    "requests": "2.32.3",
    "rich": "12.6.0",
    "uvicorn": "0.19.0",
    "fastapi": "0.115.6",
    "pycloudflared": "0.2.0",
    "ruamel.yaml": "0.17.21",
    "sqlalchemy": "2.0.52",
    "python-multipart": "0.0.6",
    "onnxruntime": "1.23.2",
    "huggingface-hub": "0.36.2",
    "wandb": "0.17.2",
    "torchsde": "0.2.6",
    "einops": "0.8.2",
    "pillow": "11.3.0",
    "imageio": "2.37.4",
    "piexif": "1.1.3",
    "picklescan": "0.0.32",
    "safetensors": "0.8.0",
    "transformers": "4.57.6",
    "sentencepiece": "0.2.2",
    "protobuf": "4.25.9",
    "tipo-kgen": "0.0.10",
    # sdkit still supplies shared model-scanning and image utilities. Native
    # inference is provided exclusively by sdkit3, so its legacy diffusion
    # dependency tree is intentionally not installed.
    "sdkit": "2.1.1",
}
modules_to_log = ["torchruntime", "torch", "torchvision", "numpy", "sdkit"]
NO_DEPENDENCY_INSTALLS = {"sdkit"}


def apply_backend_config_env_overrides(backend_config: dict, env=None) -> None:
    """
    Apply (a small allowlist of) backend_config keys from config.yaml to environment variables.

    Keeping an explicit allowlist helps prevent accidental or unsafe env overrides.
    """

    if not backend_config:
        return

    if env is None:
        env = os.environ

    def _set(key: str, value) -> None:
        if value is None:
            return
        env[key] = str(value)
        print(f"backend_config overrode {key} to {env[key]}")

    if "HSA_OVERRIDE_GFX_VERSION" in backend_config:
        _set("HSA_OVERRIDE_GFX_VERSION", backend_config["HSA_OVERRIDE_GFX_VERSION"])
    if "HIP_VISIBLE_DEVICES" in backend_config:
        _set("HIP_VISIBLE_DEVICES", backend_config["HIP_VISIBLE_DEVICES"])

    if "COMMANDLINE_ARGS" in backend_config:
        cmdline_args = backend_config["COMMANDLINE_ARGS"]
        if isinstance(cmdline_args, (list, tuple)):
            cmdline_args = " ".join(map(str, cmdline_args))
        _set("COMMANDLINE_ARGS", cmdline_args)

    if "FORCE_FULL_PRECISION" in backend_config:
        _set("FORCE_FULL_PRECISION", backend_config["FORCE_FULL_PRECISION"])


def version(module_name: str) -> str:
    try:
        return pkg_version(module_name)
    except:
        return None


def install(module_name: str, module_version: str):
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        f"{module_name}=={module_version}",
    ]
    if module_name in NO_DEPENDENCY_INSTALLS:
        install_cmd.append("--no-deps")

    print(">", " ".join(install_cmd))
    subprocess.run(install_cmd, check=True)


def update_modules():
    import torchruntime

    if version("torch") is None or version("torchvision") is None:
        torchruntime.install(["torch", "torchvision"])
    else:
        print(f"Current torch version: {version('torch')}")

    for module_name, required_version in modules_to_check.items():
        if version(module_name) == required_version:
            continue
        try:
            install(module_name, required_version)
        except subprocess.CalledProcessError:
            fail(module_name)
        if version(module_name) != required_version:
            fail(module_name)

    for module_name in modules_to_log:
        print(f"{module_name}: {version(module_name)}")


def fail(module_name):
    print(f"""Error installing {module_name}. Sorry about that, please try to:
1. Run this installer again.
2. If that doesn't fix it, please try the common troubleshooting steps at https://github.com/easydiffusion/easydiffusion/wiki/Troubleshooting
3. If those steps don't help, please copy *all* the error messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB
4. If that doesn't solve the problem, please file an issue at https://github.com/easydiffusion/easydiffusion/issues
Thanks!""")
    exit(1)


### Launcher


def get_config():
    config_directory = os.path.dirname(__file__)  # this will be "scripts"
    config_yaml = os.path.abspath(os.path.join(config_directory, "..", "config.yaml"))
    config_json = os.path.join(config_directory, "config.json")

    config = None

    # migrate the old config yaml location
    config_legacy_yaml = os.path.join(config_directory, "config.yaml")
    if os.path.isfile(config_legacy_yaml):
        shutil.move(config_legacy_yaml, config_yaml)

    if os.path.isfile(config_yaml):
        from ruamel.yaml import YAML

        yaml = YAML(typ="safe")
        with open(config_yaml, "r") as configfile:
            try:
                config = yaml.load(configfile)
            except Exception as e:
                print(e, file=sys.stderr)
    elif os.path.isfile(config_json):
        import json

        with open(config_json, "r") as configfile:
            try:
                config = json.load(configfile)
            except Exception as e:
                print(e, file=sys.stderr)

    if config is None:
        config = {}
    return config


def launch_uvicorn():
    config = get_config()

    pprint(config)

    print("\n\nEasy Diffusion installation complete, starting the server!\n\n")

    import torchruntime

    torchruntime.configure()

    # print info in a non-blocking thread
    def print_torchruntime_info():
        if hasattr(torchruntime, "info"):
            torchruntime.info()

    info_thread = threading.Thread(target=print_torchruntime_info)
    info_thread.start()

    # allow a user to override the HSA_OVERRIDE_GFX_VERSION and HIP_VISIBLE_DEVICES variables
    # until ED gets process-based multi-GPU support (which will allow different processes to use different GPUs)
    apply_backend_config_env_overrides(config.get("backend_config", {}))

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    os.environ["PYTHONPATH"] = str(Path(sys.prefix, "lib", python_version, "site-packages"))
    os.environ["SD_UI_PATH"] = str(Path(Path.cwd(), "ui"))

    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")
    print(f"Python:  {shutil.which('python')}")
    print(f"Version: {platform.python_version()}")

    bind_ip = "127.0.0.1"
    listen_port = 10000
    if "net" in config:
        print("Checking network settings")
        if "listen_port" in config["net"]:
            listen_port = config["net"]["listen_port"]
            print("Set listen port to ", listen_port)
        if "listen_to_network" in config["net"] and config["net"]["listen_to_network"] == True:
            if "bind_ip" in config["net"]:
                bind_ip = config["net"]["bind_ip"]
            else:
                bind_ip = "0.0.0.0"
            print("Set bind_ip to ", bind_ip)

    print("\nLaunching uvicorn\n")

    import uvicorn

    uvicorn.run(
        "main:server_api",
        port=listen_port,
        log_level="error",
        app_dir=os.environ["SD_UI_PATH"],
        host=bind_ip,
        access_log=False,
    )


def main():
    update_modules()

    if len(sys.argv) > 1 and sys.argv[1] == "--launch-uvicorn":
        launch_uvicorn()


if __name__ == "__main__":
    main()
