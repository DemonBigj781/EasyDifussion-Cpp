import os
import re
import threading
from pathlib import Path
from typing import Optional

from easydiffusion import app as easy_app
from easydiffusion.utils import log as _logger
from fastapi import HTTPException
from pydantic import BaseModel

from .kgen.formatter import apply_dtg_prompt, apply_format, seperate_tags
from .kgen.metainfo import TARGET
from .kgen.tipo import run_tipo
from .sidecar import load_model_settings, public_settings

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except Exception as exc:  # pragma: no cover - best-effort import
    LLAMA_AVAILABLE = False
    LLAMA_IMPORT_ERROR = exc


_env_model_dir = os.environ.get("TIPO_MODEL_DIR")
MODEL_DIR = (
    Path(_env_model_dir).expanduser().resolve()
    if _env_model_dir
    else (Path(easy_app.MODELS_DIR) / "tipo").resolve()
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
_logger.info(f"TIPO model dir: {MODEL_DIR}")

DEFAULT_DEVICE = os.environ.get("TIPO_DEVICE", "cpu")
DEFAULT_GPU_LAYERS = int(os.environ.get("TIPO_GPU_LAYERS", "0"))


SEED_MAX = 2**31 - 1

attn_syntax = (
    r"\\\\\(|"
    r"\\\\\)|"
    r"\\\\\[|"
    r"\\\\]|"
    r"\\\\\\\\|"
    r"\\\\|"
    r"\(|"
    r"\[|"
    r":\s*([+-]?[.\d]+)\s*\)|"
    r"\)|"
    r"]|"
    r"[^\\()\[\]:]+|"
    r":"
)
re_attention = re.compile(attn_syntax, re.X)
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


class GenerateRequest(BaseModel):
    tipo_model: str = ""
    tags: str = ""
    nl_prompt: str = ""
    ban_tags: str = ""
    format: Optional[str] = None
    seed: int = -1
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    tag_length: Optional[str] = None
    nl_length: Optional[str] = None
    width: int = 1024
    height: int = 1024
    device: Optional[str] = None


class GenerateResponse(BaseModel):
    formatted_prompt: str
    formatted_user_prompt: str
    unformatted_prompt: str
    unformatted_user_prompt: str
    model: str
    protocol: str
    settings: dict


_model_lock = threading.Lock()
_current_model = {"name": None, "instance": None}


def _assert_llama_available():
    if not LLAMA_AVAILABLE:
        _logger.error(f"LLAMA import failed: {LLAMA_IMPORT_ERROR}")
        raise HTTPException(
            status_code=500,
            detail=f"TIPO server missing dependencies: {LLAMA_IMPORT_ERROR}",
        )


def _parse_prompt_attention(text: str):
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for match in re_attention.finditer(text):
        piece = match.group(0)
        weight = match.group(1)

        if piece.startswith("\\"):
            res.append([piece[1:], 1.0])
        elif piece == "(":
            round_brackets.append(len(res))
        elif piece == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif piece == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif piece == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, piece)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def _apply_strength(tag_map, strength_map, strength_map_nl, break_map=None):
    if break_map is None:
        break_map = {}

    for cate in tag_map.keys():
        new_list = []
        if isinstance(tag_map[cate], str):
            if all(part in tag_map[cate] for part, _strength in strength_map_nl):
                org_prompt = tag_map[cate]
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, org_prompt = org_prompt.split(part, 1)
                    new_prompt += before.replace("(", "\\(").replace(")", "\\)")
                    part = part.replace("(", "\\(").replace(")", "\\)")
                    new_prompt += f"({part}:{strength})"
                new_prompt += org_prompt
                tag_map[cate] = new_prompt
            continue

        for org_tag in tag_map[cate]:
            tag = org_tag.replace("(", "\\(").replace(")", "\\)")
            if org_tag in strength_map:
                new_list.append(f"({tag}:{strength_map[org_tag]})")
            else:
                new_list.append(tag)
            if tag in break_map or org_tag in break_map:
                new_list.append("BREAK")
        tag_map[cate] = new_list

    return tag_map


def _clone_tag_map(tag_map):
    cloned = {}
    for key, value in tag_map.items():
        if isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _list_models():
    return sorted(
        path.relative_to(MODEL_DIR).as_posix()
        for path in MODEL_DIR.rglob("*.gguf")
        if path.is_file() and not path.is_symlink()
    )


def list_model_files():
    return [str((MODEL_DIR / model).resolve()) for model in _list_models()]


def list_model_metadata():
    metadata = []
    for model in _list_models():
        path = MODEL_DIR / model
        try:
            stat = path.stat()
        except OSError as error:
            metadata.append({"model": model, "model_name": path.name, "error": str(error)})
            continue
        settings = load_model_settings(path, strict=False)
        model_meta = {"format": "gguf"}
        if "error" in settings:
            model_meta["sidecar_error"] = settings["error"]
        else:
            model_meta.update(public_settings(settings, path))
        metadata.append(
            {
                "model": model,
                "model_name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime_ns / 1_000_000,
                "meta": model_meta,
            }
        )
    return metadata


def selector_models():
    models = []
    for model in _list_models():
        settings = load_model_settings(MODEL_DIR / model, strict=False)
        protocol = settings.get("protocol", "tipo")
        models.append({"model": model, "name": model, "tags": ["tipo", protocol]})
    return models


def _resolve_model_path(tipo_model: str) -> Path:
    candidate = tipo_model or ""
    if " | " in candidate:
        _prefix, candidate = candidate.split(" | ", 1)
    if not candidate:
        models = _list_models()
        if not models:
            raise HTTPException(status_code=404, detail="No GGUF models found.")
        candidate = models[0]
    path = (MODEL_DIR / candidate).resolve()
    try:
        path.relative_to(MODEL_DIR)
    except ValueError as error:
        raise HTTPException(status_code=403, detail="TIPO model path is outside the model directory.") from error
    if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".gguf":
        return path
    raise HTTPException(status_code=404, detail=f"Model not found: {candidate}")


def _load_model(tipo_model: str, device: str, settings):
    model_path = _resolve_model_path(tipo_model)
    if device not in ("cpu", "cuda"):
        raise HTTPException(status_code=400, detail=f"Unsupported TIPO device: {device}")
    model_key = (str(model_path), device, settings["context_length"])
    if model_key == _current_model["name"] and _current_model["instance"] is not None:
        return _current_model["instance"]

    n_gpu_layers = DEFAULT_GPU_LAYERS if device == "cuda" else 0
    old_model = _current_model["instance"]
    if old_model is not None:
        _current_model["name"] = None
        _current_model["instance"] = None
        close = getattr(old_model, "close", None)
        if close:
            close()
    _logger.info(f"Loading TIPO model: {model_path} (device={device})")
    model = Llama(
        model_path=str(model_path),
        n_ctx=settings["context_length"],
        n_gpu_layers=n_gpu_layers,
        seed=0,
        verbose=False,
    )
    _current_model["name"] = model_key
    _current_model["instance"] = model
    return model


def _black_list_match(tag, black_list):
    for pattern in black_list:
        if re.match(pattern, tag):
            return True
    return False


def _generate_tags(
    model,
    prompt,
    prompt_tags,
    len_target,
    black_list,
    temperature,
    top_p,
    min_p,
    top_k,
    seed,
    repeat_penalty=1.0,
    max_new_tokens=256,
    max_retry=5,
    max_same_output=5,
):
    llm_gen = ""
    previous_output = None
    same_output_count = 0
    for retry in range(max_retry + 1):
        try:
            result = model.create_completion(
                prompt,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                max_tokens=max_new_tokens,
                repeat_penalty=repeat_penalty,
                seed=(seed + retry) if seed is not None else retry,
            )
        except TypeError:
            _logger.warning("llama_cpp missing min_p support; retrying without it")
            result = model.create_completion(
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_new_tokens,
                repeat_penalty=repeat_penalty,
                seed=(seed + retry) if seed is not None else retry,
            )
        llm_gen = prompt + result["choices"][0]["text"]
        extra = llm_gen.split("<|input_end|>")[-1].strip().strip(",")
        raw_tokens = [tok.strip() for tok in extra.split(",") if tok.strip()]
        extra_tokens = list(
            dict.fromkeys(tok for tok in raw_tokens if not _black_list_match(tok, black_list))
        )
        llm_gen = llm_gen.replace(extra, ", ".join(extra_tokens))

        if len(prompt_tags) + len(extra_tokens) >= len_target:
            break
        output_key = tuple(extra_tokens)
        if output_key == previous_output:
            same_output_count += 1
            if same_output_count >= max_same_output:
                break
        else:
            same_output_count = 0
            previous_output = output_key
        original = llm_gen.split("<|input_end|>", 1)[0]
        prompt = f"{original}<|input_end|>{', '.join(extra_tokens)}".replace("  <|", " <|")

    return llm_gen, extra_tokens


def health():
    if not LLAMA_AVAILABLE:
        return {"status": "error", "detail": str(LLAMA_IMPORT_ERROR)}
    return {"status": "ok", "models": len(_list_models()), "model_dir": str(MODEL_DIR)}


def list_models():
    models = _list_models()
    _logger.info(f"TIPO model list requested: {len(models)} models in {MODEL_DIR}")
    metadata = {}
    for model in models:
        settings = load_model_settings(MODEL_DIR / model, strict=False)
        if "error" in settings:
            metadata[model] = {"error": settings["error"]}
        else:
            metadata[model] = public_settings(settings, MODEL_DIR / model)
    return {"models": models, "metadata": metadata}


def _effective_settings(req, model_path):
    settings = load_model_settings(model_path)
    for field in ("temperature", "top_p", "min_p", "top_k", "tag_length", "nl_length", "format"):
        value = getattr(req, field)
        if value is not None:
            settings[field] = value
    settings["tag_length"] = str(settings["tag_length"]).replace(" ", "_")
    settings["nl_length"] = str(settings["nl_length"]).replace(" ", "_")
    if settings["tag_length"] not in TARGET or settings["nl_length"] not in TARGET:
        raise HTTPException(status_code=400, detail="Unsupported TIPO length target.")
    if not 0 <= float(settings["top_p"]) <= 1 or not 0 <= float(settings["min_p"]) <= 1:
        raise HTTPException(status_code=400, detail="top_p and min_p must be between 0 and 1.")
    if float(settings["temperature"]) < 0 or int(settings["top_k"]) < 0:
        raise HTTPException(status_code=400, detail="temperature and top_k cannot be negative.")
    return settings


def _complete(model, prompt, settings, seed):
    kwargs = {
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "top_k": settings["top_k"],
        "max_tokens": settings["max_new_tokens"],
        "repeat_penalty": settings["repeat_penalty"],
        "seed": seed,
    }
    if settings["min_p"]:
        kwargs["min_p"] = settings["min_p"]
    try:
        result = model.create_completion(prompt, **kwargs)
    except TypeError:
        kwargs.pop("min_p", None)
        result = model.create_completion(prompt, **kwargs)
    return prompt + result["choices"][0]["text"]


def generate(req: GenerateRequest):
    _assert_llama_available()

    model_path = _resolve_model_path(req.tipo_model)
    settings = _effective_settings(req, model_path)
    device = req.device or DEFAULT_DEVICE

    if req.seed is None or req.seed < 0:
        seed = int.from_bytes(os.urandom(4), "little") % SEED_MAX
    else:
        seed = min(req.seed, SEED_MAX)

    tag_length = settings["tag_length"]
    len_target = TARGET.get(tag_length, TARGET["long"])

    prompt_parse_strength = _parse_prompt_attention(req.tags)
    strength_map_nl = []
    nl_prompt = ""
    for part, strength in _parse_prompt_attention(req.nl_prompt or ""):
        nl_prompt += part
        if strength != 1:
            strength_map_nl.append((part, strength))
    if settings["protocol"] != "tipo" and nl_prompt.strip():
        _logger.warning(
            "Ignoring Natural Language Prompt because DanTagGen accepts tag conditioning only."
        )
        nl_prompt = ""
        strength_map_nl = []
    black_list = [tag.strip() for tag in req.ban_tags.split(",") if tag.strip()]

    all_tags = []
    strength_map = {}
    for part, strength in prompt_parse_strength:
        part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
        all_tags.extend(part_tags)
        if strength != 1:
            for tag in part_tags:
                strength_map[tag] = strength

    org_tag_map = seperate_tags(all_tags)
    aspect_ratio = req.width / req.height if req.height else 1.0
    with _model_lock:
        model = _load_model(req.tipo_model, device, settings)
        if settings["protocol"] == "tipo":
            try:
                tag_map = run_tipo(
                    lambda prompt, run_seed: _complete(model, prompt, settings, run_seed),
                    org_tag_map,
                    nl_prompt,
                    tag_length,
                    settings["nl_length"],
                    settings["format"],
                    aspect_ratio,
                    black_list,
                    seed,
                    settings["max_retry"],
                    settings["max_same_output"],
                )
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            generated_tags = tag_map.get("special", []) + tag_map.get("general", [])
            addon_tags = [tag for tag in generated_tags if tag not in all_tags]
        else:
            dtg_prompt = apply_dtg_prompt(org_tag_map, tag_length, aspect_ratio)
            _llm_gen, extra_tokens = _generate_tags(
                model,
                dtg_prompt,
                org_tag_map.get("special", []) + org_tag_map.get("general", []),
                len_target,
                black_list,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                min_p=settings["min_p"],
                top_k=settings["top_k"],
                seed=seed,
                repeat_penalty=settings["repeat_penalty"],
                max_new_tokens=settings["max_new_tokens"],
                max_retry=settings["max_retry"],
                max_same_output=settings["max_same_output"],
            )
            addon_tags = [tag for tag in extra_tokens if tag not in all_tags]
            tag_map = seperate_tags(all_tags + addon_tags)

    user_tag_map = _clone_tag_map(org_tag_map)
    user_tag_map["extended"] = nl_prompt
    user_tag_map["generated"] = ""

    formatted_user_prompt = apply_format(
        _apply_strength(user_tag_map, strength_map, strength_map_nl),
        settings["format"],
    )

    formatted_prompt_by_tipo = apply_format(
        _apply_strength(_clone_tag_map(tag_map), strength_map, strength_map_nl),
        settings["format"],
    )
    unformatted_prompt_by_user = req.tags + (f"\n{nl_prompt}" if nl_prompt else "")
    unformatted_prompt_by_tipo = req.tags
    if addon_tags:
        unformatted_prompt_by_tipo += ", " + ", ".join(addon_tags)
    generated_nl = tag_map.get("extended") or tag_map.get("generated") or nl_prompt
    if generated_nl:
        unformatted_prompt_by_tipo += f"\n{generated_nl}"

    _logger.info(f"TIPO generated prompt: tags={len(all_tags)} addon={len(addon_tags)} seed={seed}")

    return GenerateResponse(
        formatted_prompt=formatted_prompt_by_tipo,
        formatted_user_prompt=formatted_user_prompt,
        unformatted_prompt=unformatted_prompt_by_tipo,
        unformatted_user_prompt=unformatted_prompt_by_user,
        model=model_path.relative_to(MODEL_DIR).as_posix(),
        protocol=settings["protocol"],
        settings=public_settings(settings, model_path),
    )
