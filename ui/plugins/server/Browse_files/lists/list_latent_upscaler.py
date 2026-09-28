"""Latent upscaler file listings."""

from ..directory_list import list_model_files


def list_latent_upscaler_files() -> list[str]:
    return list_model_files("latent_upscaler")


__all__ = ["list_latent_upscaler_files"]
