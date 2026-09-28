"""ControlNet file listings for all configured ControlNet model families."""

from ..directory_list import list_model_files


def list_controlnet_files() -> list[str]:
    types = ("controlnet", "controlnet-union", "uni-controlnet", "controlnet-lite",
             "controlnet-lllite", "ip-adapter", "clip-vision")
    return sorted({path for model_type in types for path in list_model_files(model_type)}, key=str.casefold)


__all__ = ["list_controlnet_files"]
