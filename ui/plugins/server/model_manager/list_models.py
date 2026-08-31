import os

from sdkit.models.model_loader.embeddings import get_embedding_token
from easydiffusion.utils import log
from easydiffusion.utils.model_identifier import identify_model_type, identify_vae_latent_family

PREFILLED_MODELS = {
    "vae": [
        {"model": "ae", "name": "ae (Flux VAE fp16)", "tags": ["vae", "vae_fx"]},
    ],
    "codeformer": [
        {"model": "codeformer", "name": "CodeFormer", "tags": ["codeformer"]},
    ],
    "text-encoder": [
        {"model": "t5xxl_fp16", "name": "T5 XXL fp16", "tags": ["text-encoder"]},
        {"model": "clip_l", "name": "CLIP L", "tags": ["text-encoder"]},
        {"model": "clip_g", "name": "CLIP G", "tags": ["text-encoder"]},
    ],
    "controlnet": [
        # {"model": "control_v11p_sd15_canny", "name": "Canny (*)", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_openpose", "name": "OpenPose (*)", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_normalbae", "name": "Normal BAE (*)", "tags": ["controlnet"]},
        # {"model": "control_v11f1p_sd15_depth", "name": "Depth (*)", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_scribble", "name": "Scribble", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_softedge", "name": "Soft Edge", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_inpaint", "name": "Inpaint", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_lineart", "name": "Line Art", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15s2_lineart_anime", "name": "Line Art Anime", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_mlsd", "name": "Straight Lines", "tags": ["controlnet"]},
        # {"model": "control_v11p_sd15_seg", "name": "Segment", "tags": ["controlnet"]},
        # {"model": "control_v11e_sd15_shuffle", "name": "Shuffle", "tags": ["controlnet"]},
        # {"model": "control_v11f1e_sd15_tile", "name": "Tile", "tags": ["controlnet"]},
    ],
}


def list_files(dir, exts):
    """
    Lists files recursively in a directory filtered by extensions using os.walk().

    Args:
        dir (str): The path to the directory to search.
        exts (list): A list of file extensions (e.g., ['.txt', '.py']).

    Returns:
        list: A list of full file paths matching the criteria.
    """
    found_files = []
    normalized_exts = tuple(ext.lower() for ext in exts)

    def scan(current_dir, ancestor_targets):
        try:
            real_dir = os.path.realpath(current_dir)
            if real_dir in ancestor_targets:
                return
            entries = list(os.scandir(current_dir))
        except OSError:
            return

        ancestor_targets = ancestor_targets | {real_dir}
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=True):
                    # Follow model-directory symlinks, but stop a link that
                    # points back to any ancestor to avoid recursive cycles.
                    scan(entry.path, ancestor_targets)
                elif entry.is_file(follow_symlinks=True) and entry.name.lower().endswith(normalized_exts):
                    found_files.append(entry.path)
            except OSError:
                continue

    if os.path.isdir(dir):
        scan(dir, set())
    return found_files


def list_models_in_dirs(dirs, exts, relative_to=None):
    models = []
    seen_real_paths = set()
    for dir in dirs:
        model_paths = list_files(dir, exts)
        model_paths = [p.replace("\\", "/") for p in model_paths]
        unique_model_paths = []
        for model_path in model_paths:
            real_model_path = os.path.realpath(model_path)
            if real_model_path in seen_real_paths:
                continue
            seen_real_paths.add(real_model_path)
            unique_model_paths.append(model_path)
        model_paths = unique_model_paths
        rel_root = relative_to if relative_to is not None else dir
        model_paths = [
            {"rel_path": os.path.relpath(p, rel_root).replace("\\", "/"), "abs_path": p}
            for p in model_paths
        ]

        existing_model_paths = [e["rel_path"] for e in models]
        model_paths = [e for e in model_paths if e["rel_path"] not in existing_model_paths]

        models.extend(model_paths)

    return models


def strip_extension(rel_path, exts):
    # Sort by length descending to match longest extensions first
    exts = sorted(exts, key=len, reverse=True)
    for ext in exts:
        if rel_path.endswith(ext):
            return rel_path[: -len(ext)]
    # Fallback to os.path.splitext for unknown extensions
    return os.path.splitext(rel_path)[0]


def set_model_metadata(model_type, models, exts):
    for m in models:
        m["model"] = strip_extension(m["rel_path"], exts)
        m["name"] = m["model"]

        if model_type == "embeddings":
            dir_name, file_name = os.path.dirname(m["model"]), os.path.basename(m["model"])
            file_name = get_embedding_token(file_name)
            m["model"] = os.path.join(dir_name, file_name).replace("\\", "/")

        if model_type == "gfpgan" and "gfpgan" not in m["model"].lower():
            m["model"] = None  # will get filtered out later

        m["tags"] = [model_type]
        if model_type in ("stable-diffusion", "video"):
            try:
                sd_model_class = identify_model_type(m["abs_path"])
            except Exception as e:
                sd_model_class = None
                # log.info(f"Could not identify model type for {m['abs_path']}: {e}")
            if sd_model_class:
                m["tags"].append(sd_model_class)
        elif model_type == "vae":
            try:
                vae_family = identify_vae_latent_family(m["abs_path"])
            except Exception:
                vae_family = None
            if vae_family:
                m["tags"].append(f"vae_{vae_family}")


def strip_null_models(models):
    return [m for m in models if m["model"]]


def strip_model_paths(models):
    for m in models:
        del m["abs_path"]
        del m["rel_path"]


def include_prefilled_models(models, prefilled_models):
    model_ids = set(m["model"] for m in models)
    for m in prefilled_models:
        if m["model"] not in model_ids:
            models.append(m)


def is_native_video_model(model, dedicated_folders):
    """Keep dedicated video folders plus recognized native video weights."""
    rel_path = model["rel_path"].replace("\\", "/").lower()
    dedicated_prefixes = tuple(folder.lower().rstrip("/") + "/" for folder in dedicated_folders)
    model_class = model["tags"][1] if len(model["tags"]) > 1 else ""

    # Mochi stores its VAE variants below the same top-level directory. Only
    # its recognized diffusion checkpoint belongs in the Video Model list.
    if "/mochi/vae/" in f"/{rel_path}":
        return False
    if "/mochi/" in f"/{rel_path}":
        return model_class.startswith("mochi_")
    if rel_path.startswith(dedicated_prefixes):
        return True

    return model_class.startswith(("wan_", "ltx_video", "mochi_"))


def is_selectable_checkpoint(model):
    """Exclude companion weights stored beside a standalone video DiT."""
    rel_path = f"/{model['rel_path'].replace(chr(92), '/').lower()}"
    return "/mochi/vae/" not in rel_path and "/mochi/t5xxl/" not in rel_path


def list_models(model_types=None):
    from easydiffusion import app
    from easydiffusion.model_manager import (
        DEDICATED_VIDEO_MODEL_FOLDER_NAMES,
        LISTABLE_MODEL_TYPES,
        MODEL_EXTENSIONS,
        get_model_dirs,
    )

    models = []
    requested_types = set(model_types) if model_types is not None else set(LISTABLE_MODEL_TYPES)

    for model_type in LISTABLE_MODEL_TYPES:
        if model_type not in requested_types:
            continue
        models_dirs = get_model_dirs(model_type)
        model_extensions = MODEL_EXTENSIONS.get(model_type, [])

        relative_to = app.MODELS_DIR if model_type == "video" else None
        models_in_dirs = list_models_in_dirs(models_dirs, model_extensions, relative_to=relative_to)
        set_model_metadata(model_type, models_in_dirs, model_extensions)
        if model_type == "video":
            models_in_dirs = [
                model
                for model in models_in_dirs
                if is_native_video_model(model, DEDICATED_VIDEO_MODEL_FOLDER_NAMES)
            ]
        elif model_type == "stable-diffusion":
            models_in_dirs = [model for model in models_in_dirs if is_selectable_checkpoint(model)]
        strip_model_paths(models_in_dirs)
        models_in_dirs = strip_null_models(models_in_dirs)
        include_prefilled_models(models_in_dirs, PREFILLED_MODELS.get(model_type, []))

        models.extend(models_in_dirs)

    return models
