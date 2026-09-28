"""Text encoder file listings."""

from ..directory_list import list_model_files


def list_text_encoder_files() -> list[str]:
    return list_model_files("text-encoder")


__all__ = ["list_text_encoder_files"]
