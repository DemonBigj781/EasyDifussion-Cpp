"""Validation for user-supplied native backend command-line arguments."""

import re
import shlex
import subprocess

# Preserve the app's model-folder config keys; only startup argv changes.
LEGACY_MODEL_DIRECTORY_FLAGS = {
    "stable-diffusion": "--ckpt-dir",
    "vae": "--vae-dir",
    "hypernetwork": "--hypernetwork-dir",
    "gfpgan": "--gfpgan-models-path",
    "realesrgan": "--realesrgan-models-path",
    "lora": "--lora-dir",
    "codeformer": "--codeformer-models-path",
    "embeddings": "--embeddings-dir",
    "controlnet": "--controlnet-dir",
    "text-encoder": "--text-encoder-dir",
}
MODEL_DIRECTORY_FLAGS = {kind: f"--dir-{kind}" for kind in LEGACY_MODEL_DIRECTORY_FLAGS}
# Block both dialects so advanced args cannot override app-managed directories,
# even if the currently installed binary uses an old launcher bundle.
MANAGED_BACKEND_ARGUMENTS = {
    "--port", "--parent-pid",
    *MODEL_DIRECTORY_FLAGS.values(),
    *LEGACY_MODEL_DIRECTORY_FLAGS.values(),
}


def probe_model_directory_flags(executable, *, cwd, env):
    """Select one complete, advertised directory vocabulary from the installed sdkit."""
    try:
        probe = subprocess.run([executable, "--help"], cwd=cwd, env=env, timeout=5,
                               capture_output=True, text=True, errors="replace", check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Could not read native sdkit --help to choose model-directory flags: {error}") from error

    # Only accept option declarations at the start of help lines, not
    # spellings mentioned in diagnostic text or another option's description.
    help_text = probe.stdout + "\n" + probe.stderr
    advertised = set(re.findall(r"^[ \t]*(--[a-z][a-z0-9-]*)(?=[ \t]|$)",
                                help_text, flags=re.MULTILINE))
    for vocabulary in (MODEL_DIRECTORY_FLAGS, LEGACY_MODEL_DIRECTORY_FLAGS):
        if set(vocabulary.values()) <= advertised:
            return vocabulary
    raise RuntimeError("Native sdkit --help does not advertise a complete supported model-directory CLI.")


def parse_backend_commandline_args(value):
    """Return an argv list without invoking a shell or overriding managed flags."""
    if isinstance(value, str):
        arguments = shlex.split(value)
    elif isinstance(value, (list, tuple)):
        arguments = [str(argument) for argument in value]
    else:
        raise ValueError("Backend command-line arguments must be text or a list.")

    for index, argument in enumerate(arguments):
        option = argument.split("=", 1)[0]
        if option in MANAGED_BACKEND_ARGUMENTS:
            raise ValueError(f"{option} is managed by Easy Diffusion and cannot be injected.")
        if option in ("--help", "-h"):
            raise ValueError(f"{option} would stop the backend instead of starting it.")
        if (argument == "--devices" and index + 1 < len(arguments)
                and arguments[index + 1] == "list"):
            raise ValueError("--devices list is a standalone diagnostic and would not start the backend.")
    return arguments
