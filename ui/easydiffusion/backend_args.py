"""Validation for user-supplied native backend command-line arguments."""

import shlex


MANAGED_BACKEND_ARGUMENTS = {
    "--port",
    "--parent-pid",
    "--ckpt-dir",
    "--vae-dir",
    "--hypernetwork-dir",
    "--gfpgan-models-path",
    "--realesrgan-models-path",
    "--lora-dir",
    "--codeformer-models-path",
    "--embeddings-dir",
    "--controlnet-dir",
    "--text-encoder-dir",
}


def parse_backend_commandline_args(value):
    """Return an argv list without invoking a shell or overriding managed flags."""
    if isinstance(value, str):
        arguments = shlex.split(value)
    elif isinstance(value, (list, tuple)):
        arguments = [str(argument) for argument in value]
    else:
        raise ValueError("Backend command-line arguments must be text or a list.")

    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in MANAGED_BACKEND_ARGUMENTS:
            raise ValueError(f"{option} is managed by Easy Diffusion and cannot be injected.")
        if option in ("--help", "-h"):
            raise ValueError(f"{option} would stop the backend instead of starting it.")
    return arguments
