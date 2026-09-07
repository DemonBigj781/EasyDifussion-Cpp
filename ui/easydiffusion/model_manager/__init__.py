import os
from glob import glob
from typing import Union
from os import path

from easydiffusion import app
from easydiffusion.types import ModelsData
from easydiffusion.utils import log
from sdkit import Context
from sdkit.models import scan_model, download_model, get_model_info_from_db
from sdkit.utils import hash_file_quick
from .list_models import list_models

KNOWN_MODEL_TYPES = [
    "stable-diffusion",
    "vae",
    "taesdvae",
    "hypernetwork",
    "gfpgan",
    "realesrgan",
    "lora",
    "codeformer",
    "embeddings",
    "controlnet",
    "controlnet-union",
    "uni-controlnet",
    "controlnet-lite",
    "controlnet-lllite",
    "ip-adapter",
    "clip-vision",
    "latent-interposer",
    "furception-vae",
    "wd14-tagger",
    "text-encoder",
]
LISTABLE_MODEL_TYPES = KNOWN_MODEL_TYPES + ["video"]
MODEL_EXTENSIONS = {
    "stable-diffusion": [".ckpt", ".safetensors", ".sft", ".gguf"],
    "video": [".ckpt", ".safetensors", ".sft", ".gguf"],
    "vae": [".vae.pt", ".ckpt", ".safetensors", ".sft", ".gguf"],
    "taesdvae": [".pt", ".pth", ".ckpt", ".safetensors", ".sft", ".gguf"],
    "hypernetwork": [".pt", ".safetensors", ".sft"],
    "gfpgan": [".pth"],
    "realesrgan": [".pth"],
    "lora": [".ckpt", ".safetensors", ".sft", ".pt"],
    "codeformer": [".pth"],
    "embeddings": [".pt", ".bin", ".safetensors", ".sft"],
    "controlnet": [".pth", ".ckpt", ".safetensors", ".sft"],
    "controlnet-union": [".pth", ".ckpt", ".safetensors", ".sft"],
    "uni-controlnet": [".pth", ".ckpt", ".safetensors", ".sft"],
    "controlnet-lite": [".pth", ".ckpt", ".safetensors", ".sft"],
    "controlnet-lllite": [".safetensors", ".sft"],
    "ip-adapter": [".safetensors", ".sft"],
    "clip-vision": [".safetensors", ".sft", ".bin"],
    "latent-interposer": [".safetensors", ".sft"],
    "furception-vae": [".safetensors", ".sft"],
    "wd14-tagger": [".onnx"],
    "text-encoder": [".safetensors", ".sft", ".gguf"],
}
DEFAULT_MODELS = {
    "stable-diffusion": [
        {"file_name": "sd-v1-5.safetensors", "model_id": "1.5-pruned-emaonly-fp16"},
    ],
    "gfpgan": [
        {"file_name": "GFPGANv1.4.pth", "model_id": "1.4"},
    ],
    "realesrgan": [
        {"file_name": "RealESRGAN_x4plus.pth", "model_id": "x4plus"},
        {"file_name": "RealESRGAN_x4plus_anime_6B.pth", "model_id": "x4plus_anime_6"},
    ],
    "vae": [
        {"file_name": "vae-ft-mse-840000-ema-pruned.ckpt", "model_id": "vae-ft-mse-840000-ema-pruned"},
    ],
}
MODELS_TO_LOAD_ON_START = ["stable-diffusion", "vae", "hypernetwork", "lora"]
ALTERNATE_FOLDER_NAMES = {  # for WebUI compatibility
    # The old stable-diffusion alias is no longer authoritative; checkpoints/
    # is the canonical shared-store hierarchy.
    "stable-diffusion": "checkpoints",
    "vae": "VAE",
    "taesdvae": "VAE-taesd",
    "hypernetwork": "hypernetworks",
    "codeformer": "Codeformer",
    "gfpgan": "GFPGAN",
    "realesrgan": "RealESRGAN",
    "lora": "Lora",
    "controlnet": "ControlNet",
    "controlnet-union": "Controlnet_Union",
    "uni-controlnet": "Uni_Controlnet",
    "controlnet-lite": "Controlnet_LITE",
    "controlnet-lllite": "controlnetLLLite",
    "ip-adapter": "ipadapter",
    "clip-vision": "clip_vision",
    # Keep the recovered staged-interposer models in the user's established
    # shared model folder.
    "latent-interposer": "interpose",
    "furception-vae": "vae/Furception",
    "wd14-tagger": "deepdanbooru",
    "text-encoder": "text_encoder",
}

known_models = {}

CONTROLNET_CATEGORY_FOLDERS = {
    "Controlnet_Union",
    "Uni_Controlnet",
    "Controlnet_LITE",
}

PREFER_ALTERNATE_FOLDER_TYPES = {
    "stable-diffusion",
    "controlnet-union",
    "uni-controlnet",
    "controlnet-lite",
    "taesdvae",
}

MODEL_DIRECTORY_CONFIG_KEYS = {
    "stable-diffusion": "checkpoints",
}

DIRECTORY_MODEL_TYPES = tuple(KNOWN_MODEL_TYPES) + ("video", "tipo")
DIRECTORY_CONFIG_KEYS = tuple(
    dict.fromkeys(MODEL_DIRECTORY_CONFIG_KEYS.get(model_type, model_type) for model_type in DIRECTORY_MODEL_TYPES)
)

# Native stable-diffusion.cpp video weights are commonly stored outside the
# image checkpoint hierarchy. Keep a separate selector index while accepting
# both Easy Diffusion and ComfyUI-style layouts.
DEDICATED_VIDEO_MODEL_FOLDER_NAMES = (
    "checkpoints/video",
    "video",
    "SVD",
    "svd",
    "lighttricks",
    "LTXVideo",
    "ltx-video",
    "Wan",
    "wan",
    "mochi",
    "Mochi",
)
VIDEO_MODEL_FOLDER_NAMES = DEDICATED_VIDEO_MODEL_FOLDER_NAMES + (
    "diffusion_models",
    "DiffusionModels",
)

VIDEO_COMPANION_MODEL_FOLDER_NAMES = {
    "vae": ("mochi/vae", "Mochi/vae"),
    "text-encoder": (
        "Text-encoder",
        "Text_Encoder",
        "mochi/t5xxl",
        "Mochi/t5xxl",
    ),
}

# Native image checkpoints converted to GGUF are kept out of checkpoints/ so
# the original and converted files can share the same architecture-relative
# name without colliding. The model selector still treats this as another
# stable-diffusion checkpoint root and scans its architecture folders.
IMAGE_GGUF_MODEL_FOLDER_NAMES = (
    "Image_GGUF",
    "image_gguf",
)

AUTOMATIC_CONTROLNET_MODEL = "__automatic_uni_union__"


def init():
    make_model_folders()


def load_default_models(context: Context):
    from easydiffusion import runtime
    from easydiffusion.backend_manager import backend

    runtime.set_vram_optimizations(context)
    configured_models = app.getConfig().get("model", {})

    # init default model paths
    for model_type in MODELS_TO_LOAD_ON_START:
        # A persisted null checkpoint is the explicit Image Settings "None"
        # selection. Do not replace it with the bundled default at startup;
        # native video can load its independently selected checkpoint later.
        if (
            model_type == "stable-diffusion"
            and model_type in configured_models
            and configured_models[model_type] is None
        ):
            context.model_paths[model_type] = None
            continue
        context.model_paths[model_type] = resolve_model_to_use(model_type=model_type, fail_if_not_found=False)
        try:
            backend.load_model(
                context,
                model_type,
                scan_model=context.model_paths[model_type] != None
                and not context.model_paths[model_type].endswith(".safetensors"),
            )
            if hasattr(context, "model_load_errors") and model_type in context.model_load_errors:
                del context.model_load_errors[model_type]
        except Exception as e:
            log.error(f"[red]Error while loading {model_type} model: {context.model_paths[model_type]}[/red]")
            if "DefaultCPUAllocator: not enough memory" in str(e):
                log.error(
                    f"[red]Your PC is low on system RAM. Please add some virtual memory (or swap space) by following the instructions at this link: https://www.ibm.com/docs/en/opw/8.2.0?topic=tuning-optional-increasing-paging-file-size-windows-computers[/red]"
                )
            else:
                log.exception(e)
            del context.model_paths[model_type]

            if not hasattr(context, "model_load_errors"):
                context.model_load_errors = {}
            context.model_load_errors[model_type] = str(e)  # storing the entire Exception can lead to memory leaks


def unload_all(context: Context):
    from easydiffusion.backend_manager import backend

    for model_type in KNOWN_MODEL_TYPES:
        backend.unload_model(context, model_type)
        if hasattr(context, "model_load_errors") and model_type in context.model_load_errors:
            del context.model_load_errors[model_type]


def resolve_model_to_use(model_name: Union[str, list] = None, model_type: str = None, fail_if_not_found: bool = True):
    model_names = model_name if isinstance(model_name, list) else [model_name]
    model_paths = []
    for m in model_names:
        if model_type == "embeddings":
            resolved = resolve_model_to_use_single(m, model_type, False)
            if resolved is None:
                resolved = resolve_embedding_reference(m)
            if resolved is not None:
                model_paths.append(resolved)
                continue

        path = resolve_model_to_use_single(m, model_type, fail_if_not_found)
        model_paths.append(path)

    return model_paths[0] if len(model_paths) == 1 else model_paths


def _embedding_reference_key(model_name: str) -> str:
    """Match UI embedding tokens without losing real filename underscores."""
    normalized = str(model_name).replace("\\", "/").strip("/")
    for extension in sorted(MODEL_EXTENSIONS["embeddings"], key=len, reverse=True):
        if normalized.lower().endswith(extension.lower()):
            normalized = normalized[: -len(extension)]
            break

    parts = []
    for part in normalized.split("/"):
        # get_embedding_token() maps spaces to underscores. Treat either as a
        # separator for lookup while retaining the exact path returned below.
        parts.append("_".join(part.replace("_", " ").split()).casefold())
    return "/".join(parts)


def resolve_embedding_reference(model_name: str):
    if not model_name:
        return None

    requested_key = _embedding_reference_key(model_name)
    match_basename_only = "/" not in str(model_name).replace("\\", "/").strip("/")
    matches = []
    extensions = tuple(extension.casefold() for extension in MODEL_EXTENSIONS["embeddings"])

    for model_dir in get_model_dirs("embeddings"):
        for root, _, files in os.walk(model_dir):
            for filename in files:
                if not filename.casefold().endswith(extensions):
                    continue
                candidate = os.path.join(root, filename)
                relative = os.path.relpath(candidate, model_dir).replace("\\", "/")
                candidate_reference = os.path.basename(relative) if match_basename_only else relative
                if _embedding_reference_key(candidate_reference) == requested_key:
                    matches.append(candidate)

    if not matches:
        return None
    matches.sort()
    if len(matches) > 1:
        log.warn(f"Multiple embeddings match {model_name}; using {matches[0]}")
    return matches[0]


def resolve_model_to_use_single(model_name: str = None, model_type: str = None, fail_if_not_found: bool = True):
    model_extensions = MODEL_EXTENSIONS.get(model_type, [])
    default_models = DEFAULT_MODELS.get(model_type, [])
    config = app.getConfig()

    if not model_name:  # When None try user configured model.
        # config = getConfig()
        if "model" in config and model_type in config["model"]:
            model_name = config["model"][model_type]

    # Dedicated model selectors return paths relative to models_dir (for
    # example SVD/svd_xt_1_1). Resolve those paths before trying the image
    # checkpoint roots, while preventing traversal outside the model store.
    if model_type == "stable-diffusion" and model_name:
        normalized_name = str(model_name).replace("\\", "/").lstrip("/")
        models_root = os.path.realpath(app.MODELS_DIR)
        candidate_stem = os.path.realpath(os.path.join(models_root, normalized_name))
        try:
            confined = os.path.commonpath([models_root, candidate_stem]) == models_root
        except ValueError:
            confined = False
        if confined:
            for candidate in [candidate_stem] + [candidate_stem + ext for ext in model_extensions]:
                if os.path.isfile(candidate):
                    return candidate

    # Dedicated native ControlNet panels prefix their selections with the
    # top-level architecture folder. Resolve those paths from models_dir while
    # keeping the request's established `controlnet` model type.
    if model_type == "controlnet" and model_name:
        normalized_name = str(model_name).replace("\\", "/").lstrip("/")
        category, separator, relative_name = normalized_name.partition("/")
        if separator and category in CONTROLNET_CATEGORY_FOLDERS and relative_name:
            category_root = os.path.realpath(os.path.join(app.MODELS_DIR, category))
            candidate_stem = os.path.realpath(os.path.join(category_root, relative_name))
            try:
                confined = os.path.commonpath([category_root, candidate_stem]) == category_root
            except ValueError:
                confined = False
            if confined:
                for candidate in [candidate_stem] + [candidate_stem + ext for ext in model_extensions]:
                    if os.path.isfile(candidate):
                        return candidate

    for model_dir in get_model_dirs(model_type):
        if model_name:
            # Check models directory
            model_path = os.path.join(model_dir, model_name)
            if os.path.exists(model_path):
                return model_path
            for model_extension in model_extensions:
                if os.path.exists(model_path + model_extension):
                    return model_path + model_extension
                if os.path.exists(model_name + model_extension):
                    return os.path.abspath(model_name + model_extension)

            # Easy Diffusion configurations commonly store only a model stem,
            # while shared model stores group checkpoints into family folders
            # such as sdxl/ and 1.5/. Resolve that unambiguous basename here.
            requested_names = {os.path.basename(model_name)}
            requested_names.update(os.path.basename(model_name) + ext for ext in model_extensions)
            recursive_matches = []
            for root, _, files in os.walk(model_dir):
                recursive_matches.extend(os.path.join(root, name) for name in files if name in requested_names)
            if recursive_matches:
                recursive_matches.sort()
                if len(recursive_matches) > 1:
                    log.warn(
                        f"Multiple {model_type} models match {model_name}; using {recursive_matches[0]}"
                    )
                return recursive_matches[0]

        # Can't find requested model, check the default paths.
        if model_type == "stable-diffusion" and not fail_if_not_found:
            for default_model in default_models:
                default_model_path = os.path.join(model_dir, default_model["file_name"])
                if os.path.exists(default_model_path):
                    if model_name is not None:
                        log.warn(
                            f"Could not find the configured custom model {model_name}. Using the default one: {default_model_path}"
                        )
                    return default_model_path

    if model_name and fail_if_not_found:
        raise FileNotFoundError(
            f"Could not find the desired model {model_name}! Is it present in the {model_dir} folder?"
        )


def reload_models_if_necessary(context: Context, models_data: ModelsData, models_to_force_reload: list = []):
    from easydiffusion.backend_manager import backend

    models_to_reload = {
        model_type: path
        for model_type, path in models_data.model_paths.items()
        if context.model_paths.get(model_type) != path or (path is not None and context.models.get(model_type) is None)
    }

    if models_data.model_paths.get("codeformer"):
        if "realesrgan" not in models_to_reload and "realesrgan" not in context.models:
            default_realesrgan = DEFAULT_MODELS["realesrgan"][0]["file_name"]
            models_to_reload["realesrgan"] = resolve_model_to_use(default_realesrgan, "realesrgan")
        elif "realesrgan" in models_to_reload and models_to_reload["realesrgan"] is None:
            del models_to_reload["realesrgan"]  # don't unload realesrgan

    for model_type in models_to_force_reload:
        if model_type not in models_data.model_paths:
            continue
        models_to_reload[model_type] = models_data.model_paths[model_type]

    for model_type, model_path_in_req in models_to_reload.items():
        context.model_paths[model_type] = model_path_in_req

        action_fn = backend.unload_model if context.model_paths[model_type] is None else backend.load_model
        extra_params = models_data.model_params.get(model_type, {})
        try:
            action_fn(context, model_type, scan_model=False, **extra_params)  # we've scanned them already
            if hasattr(context, "model_load_errors") and model_type in context.model_load_errors:
                del context.model_load_errors[model_type]
        except Exception as e:
            log.exception(e)
            if action_fn == backend.load_model:
                if not hasattr(context, "model_load_errors"):
                    context.model_load_errors = {}
                context.model_load_errors[model_type] = str(e)  # storing the entire Exception can lead to memory leaks

    if hasattr(backend, "flush_model_changes"):
        backend.flush_model_changes(context)


def resolve_model_paths(models_data: ModelsData):
    from easydiffusion.backend_manager import backend

    cn_filters = backend.list_controlnet_filters()

    model_paths = models_data.model_paths
    skip_models = cn_filters + [
        "latent_upscaler",
        "nsfw_checker",
        "esrgan_4x",
        "lanczos",
        "nearest",
        "scunet",
        "swinir",
    ]

    for model_type in model_paths:
        if model_type in skip_models:  # doesn't use model paths
            continue

        # Automatic Uni/Union is a native backend routing sentinel, not a
        # checkpoint filename. Keep it intact for the generation request.
        if model_type == "controlnet" and model_paths[model_type] == AUTOMATIC_CONTROLNET_MODEL:
            continue

        # ModelsData uses None to request an unload. In particular, the image
        # model selector can now be left at None while another task (such as
        # native video) supplies its own checkpoint.
        if model_paths[model_type] is None:
            continue

        if model_type in ("vae", "codeformer", "controlnet", "text-encoder") and model_paths[model_type]:
            model_ids = model_paths[model_type]
            model_ids = model_ids if isinstance(model_ids, list) else [model_ids]

            new_model_paths = []

            for model_id in model_ids:
                # log.info(f"Checking for {model_id=}")
                model_info = get_model_info_from_db(model_type=model_type, model_id=model_id)
                if model_info:
                    filename = model_info.get("url", "").split("/")[-1]
                    download_if_necessary(model_type, filename, model_id, skip_if_others_exist=False)

                    new_model_paths.append(path.splitext(filename)[0])
                else:  # not in the model db, probably a regular file
                    new_model_paths.append(model_id)

            model_paths[model_type] = new_model_paths

        model_paths[model_type] = resolve_model_to_use(model_paths[model_type], model_type=model_type)


def fail_if_models_did_not_load(context: Context):
    for model_type in KNOWN_MODEL_TYPES:
        if hasattr(context, "model_load_errors") and model_type in context.model_load_errors:
            e = context.model_load_errors[model_type]
            raise Exception(f"Could not load the {model_type} model! Reason: " + e)


def download_if_necessary(model_type: str, file_name: str, model_id: str, skip_if_others_exist=True):
    from easydiffusion.backend_manager import backend

    expected_hash = get_model_info_from_db(model_type=model_type, model_id=model_id)["quick_hash"]
    other_models_exist = any_model_exists(model_type) and skip_if_others_exist

    for model_dir in get_model_dirs(model_type):
        model_path = os.path.join(model_dir, file_name)

        known_model_exists = os.path.exists(model_path)
        known_model_is_corrupt = known_model_exists and hash_file_quick(model_path) != expected_hash

        needs_download = known_model_is_corrupt or (not other_models_exist and not known_model_exists)

        # log.info(f"{model_path=} {needs_download=}")
        # if known_model_exists:
        #     log.info(f"{expected_hash=} {hash_file_quick(model_path)=}")
        # log.info(f"{known_model_is_corrupt=} {other_models_exist=} {known_model_exists=}")

        if not needs_download:
            return

    print("> download", model_type, model_id)
    download_model(model_type, model_id, download_base_dir=app.MODELS_DIR, download_config_if_available=False)

    backend.refresh_models()


def any_model_exists(model_type: str) -> bool:
    extensions = MODEL_EXTENSIONS.get(model_type, [])
    for model_dir in get_model_dirs(model_type):
        for ext in extensions:
            if any(glob(f"{model_dir}/**/*{ext}", recursive=True)):
                return True

    return False


def make_model_folders():
    for model_type in KNOWN_MODEL_TYPES:
        model_dir_path = get_model_dirs(model_type)[0]

        try:
            os.makedirs(model_dir_path, exist_ok=True)
        except Exception as e:
            from rich.console import Console
            from rich.panel import Panel

            Console().print(
                Panel(
                    "\n"
                    + f"Error while creating the models directory: '{model_dir_path}'\n"
                    + f"Error: {e}\n\n"
                    + f"[white]Check the 'models_dir:' line in the file '{os.path.join(app.ROOT_DIR, 'config.yaml')}'.[/white]\n",
                    title="Fatal Error starting Easy Diffusion",
                    style="bold yellow on red",
                )
            )
            input("Press Enter to terminate...")
            exit(1)

        help_file_name = f"Place your {model_type} model files here.txt"
        help_file_contents = f'Supported extensions: {" or ".join(MODEL_EXTENSIONS.get(model_type))}'
        try:
            with open(os.path.join(model_dir_path, help_file_name), "w", encoding="utf-8") as f:
                f.write(help_file_contents)
        except Exception as e:
            log.exception(e)


def is_malicious_model(file_path):
    try:
        if file_path.endswith((".safetensors", ".sft", ".gguf")):
            return False
        scan_result = scan_model(file_path)
        if scan_result.issues_count > 0 or scan_result.infected_files > 0:
            log.warn(
                ":warning: [bold red]Scan %s: %d scanned, %d issue, %d infected.[/bold red]"
                % (
                    file_path,
                    scan_result.scanned_files,
                    scan_result.issues_count,
                    scan_result.infected_files,
                )
            )
            return True
        else:
            log.debug(
                "Scan %s: [green]%d scanned, %d issue, %d infected.[/green]"
                % (
                    file_path,
                    scan_result.scanned_files,
                    scan_result.issues_count,
                    scan_result.infected_files,
                )
            )
            return False
    except Exception as e:
        log.error(f"error while scanning: {file_path}, error: {e}")
    return False


def normalize_directory_config(value):
    """Validate the per-model directory map stored in config.yaml."""
    if not isinstance(value, dict):
        raise ValueError("directories must be a mapping of model family names to paths")

    normalized = {}
    for key, raw_paths in value.items():
        if key not in DIRECTORY_CONFIG_KEYS:
            raise ValueError(f"Unknown model directory family: {key}")
        paths = raw_paths if isinstance(raw_paths, (list, tuple)) else [raw_paths]
        clean_paths = []
        for raw_path in paths:
            if raw_path is None or not str(raw_path).strip():
                continue
            path_value = str(raw_path).strip()
            if "\x00" in path_value:
                raise ValueError(f"Invalid path for model directory family: {key}")
            if path_value not in clean_paths:
                clean_paths.append(path_value)
        if clean_paths:
            normalized[key] = clean_paths[0] if len(clean_paths) == 1 else clean_paths
    return normalized


def _configured_model_dirs(model_type, base_dir):
    directory_key = MODEL_DIRECTORY_CONFIG_KEYS.get(model_type, model_type)
    directories = app.getConfig().get("directories") or {}
    raw_paths = directories.get(directory_key)
    if raw_paths is None:
        return []
    paths = raw_paths if isinstance(raw_paths, (list, tuple)) else [raw_paths]
    resolved = []
    for raw_path in paths:
        if raw_path is None or not str(raw_path).strip():
            continue
        candidate = os.path.expanduser(str(raw_path).strip())
        if not os.path.isabs(candidate):
            candidate = os.path.join(base_dir, candidate)
        candidate = os.path.abspath(candidate)
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def _append_unique_directories(directories, candidates):
    seen = {os.path.realpath(candidate) for candidate in directories}
    for candidate in candidates:
        real_candidate = os.path.realpath(candidate)
        if real_candidate not in seen:
            seen.add(real_candidate)
            directories.append(candidate)


def _existing_image_gguf_dirs(base_dir):
    return [
        os.path.join(base_dir, folder_name)
        for folder_name in IMAGE_GGUF_MODEL_FOLDER_NAMES
        if os.path.isdir(os.path.join(base_dir, folder_name))
    ]


def _existing_video_companion_dirs(model_type, base_dir):
    return [
        os.path.join(base_dir, folder_name)
        for folder_name in VIDEO_COMPANION_MODEL_FOLDER_NAMES.get(model_type, ())
        if os.path.isdir(os.path.join(base_dir, folder_name))
    ]


def get_model_dirs(model_type: str, base_dir=None):
    "Returns the possible model directory paths for the given model type. Mainly used for WebUI compatibility"

    if base_dir is None:
        base_dir = app.MODELS_DIR

    configured_dirs = _configured_model_dirs(model_type, base_dir)
    if configured_dirs:
        if model_type == "stable-diffusion":
            _append_unique_directories(configured_dirs, _existing_image_gguf_dirs(base_dir))
        if model_type == "controlnet":
            _append_unique_directories(configured_dirs, _configured_model_dirs("controlnet-lite", base_dir))
        _append_unique_directories(configured_dirs, _existing_video_companion_dirs(model_type, base_dir))
        return configured_dirs

    if model_type == "video":
        dirs = []
        seen = set()
        for folder_name in VIDEO_MODEL_FOLDER_NAMES:
            candidate = os.path.join(base_dir, folder_name)
            if not os.path.isdir(candidate):
                continue
            real_candidate = os.path.realpath(candidate)
            if real_candidate in seen:
                continue
            seen.add(real_candidate)
            dirs.append(candidate)
        return dirs or [os.path.join(base_dir, DEDICATED_VIDEO_MODEL_FOLDER_NAMES[0])]

    primary_dir = os.path.join(base_dir, model_type)
    dirs = [primary_dir]

    if model_type in ALTERNATE_FOLDER_NAMES:
        alt_dir = ALTERNATE_FOLDER_NAMES[model_type]
        alt_dir = os.path.join(base_dir, alt_dir)
        if (
            os.path.exists(alt_dir)
            and os.path.isdir(alt_dir)
            and os.path.realpath(primary_dir) != os.path.realpath(alt_dir)
        ):
            if model_type in PREFER_ALTERNATE_FOLDER_TYPES:
                # Prefer explicit canonical shared-store folders. Do not
                # create duplicate lower-case architecture directories; a
                # genuinely distinct legacy folder remains a secondary lookup
                # location when it already exists.
                dirs = [alt_dir]
                if os.path.exists(primary_dir):
                    dirs.append(primary_dir)
            else:
                dirs.append(alt_dir)

    # Video companion weights stay separate from Image Settings while using
    # the normal VAE/text-encoder model types in the render request.
    _append_unique_directories(dirs, _existing_video_companion_dirs(model_type, base_dir))

    # ControlNet-LITE uses the standard ControlNet request path with a
    # different backbone, so expose its directory in the standard selector.
    if model_type == "controlnet":
        lite_dirs = _configured_model_dirs("controlnet-lite", base_dir)
        if not lite_dirs:
            lite_dirs = [
                os.path.join(base_dir, folder_name)
                for folder_name in ("Controlnet_LITE", "controlnet-lite")
                if os.path.isdir(os.path.join(base_dir, folder_name))
            ]
        _append_unique_directories(dirs, lite_dirs)

    if model_type == "stable-diffusion":
        _append_unique_directories(dirs, _existing_image_gguf_dirs(base_dir))

    return dirs


def effective_directory_config(base_dir=None):
    """Return resolved directory defaults for the System Settings editor."""
    if base_dir is None:
        base_dir = app.MODELS_DIR
    result = {}
    for model_type in DIRECTORY_MODEL_TYPES:
        key = MODEL_DIRECTORY_CONFIG_KEYS.get(model_type, model_type)
        directories = get_model_dirs(model_type, base_dir)
        result[key] = directories[0] if len(directories) == 1 else directories
    return result
