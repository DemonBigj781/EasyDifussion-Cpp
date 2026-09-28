"""Upscaler file listings."""

from ..directory_list import list_model_files


def list_upscaler_files() -> list[str]:
    return list_model_files("realesrgan")


__all__ = ["list_upscaler_files"]
