"""Latent interposer file listings."""

from ..directory_list import list_model_files


def list_latent_interpose_files() -> list[str]:
    return list_model_files("latent-interposer")


__all__ = ["list_latent_interpose_files"]
