import logging
import os
import re
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_log_env = os.environ.get("TIPO_LOG_FILE")
if _log_env:
    LOG_PATH = Path(_log_env).expanduser()
else:
    LOG_PATH = Path(__file__).resolve().parents[1] / "tipo_plugin.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("tipo_plugin")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

try:
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel
except Exception as exc:
    _logger.error("Missing server dependencies: %s", exc)
    raise

from .kgen.formatter import apply_dtg_prompt, apply_format, seperate_tags
from .cors import apply_cors
from .kgen.metainfo import TARGET

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except Exception as exc:  # pragma: no cover - best-effort import
    LLAMA_AVAILABLE = False
    LLAMA_IMPORT_ERROR = exc


def get_logger():
    return _logger


APP = FastAPI()
apply_cors(APP)


def _find_ed_root():
    for start in (Path.cwd(), Path(__file__).absolute()):
        for parent in start.parents:
            if (parent / "installer_files").exists() and (parent / "plugins").exists():
                return parent
    return None


def _load_config():
    config_path = Path(__file__).resolve().parent / "config.json"
    if not config_path.exists():
        return {}


def _resolve_model_dir(path_value: str, ed_root: Optional[Path]):
    path = Path(path_value)
    if path.is_absolute():
        return path
    if ed_root is not None:
        candidate = ed_root / path
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()
    try:
        import json

        raw = config_path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        _logger.warning("Failed to load config.json: %s", exc)
        return {}


_config = _load_config()

_env_model_dir = os.environ.get("TIPO_MODEL_DIR")
_cfg_model_dir = _config.get("model_dir") if isinstance(_config, dict) else None

ed_root = _find_ed_root()

if _env_model_dir:
    MODEL_DIR = Path(_env_model_dir)
elif _cfg_model_dir:
    MODEL_DIR = _resolve_model_dir(_cfg_model_dir, ed_root)
else:
    if ed_root and (ed_root / "models" / "tipo").exists():
        MODEL_DIR = ed_root / "models" / "tipo"
    else:
        MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
_logger.info("TIPO model dir: %s", MODEL_DIR)

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
    tipo_model: str
    tags: str = ""
    nl_prompt: str = ""
    ban_tags: str = ""
    format: str = ""
    seed: int = -1
    temperature: float = 0.5
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 80
    tag_length: str = "long"
    nl_length: str = "long"
    width: int = 1024
    height: int = 1024
    device: str = DEFAULT_DEVICE


class GenerateResponse(BaseModel):
    formatted_prompt: str
    formatted_user_prompt: str
    unformatted_prompt: str
    unformatted_user_prompt: str


_model_lock = threading.Lock()
_current_model = {"name": None, "instance": None}


def _assert_llama_available():
    if not LLAMA_AVAILABLE:
        _logger.error("LLAMA import failed: %s", LLAMA_IMPORT_ERROR)
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


def _maybe_use_cwd_models():
    global MODEL_DIR
    cwd = Path.cwd()
    candidate = cwd / "models" / "tipo"
    if candidate.exists() and any(candidate.glob("*.gguf")):
        if candidate != MODEL_DIR:
            MODEL_DIR = candidate
            _logger.info("Using TIPO models from cwd: %s", MODEL_DIR)


def _list_models():
    models = sorted([path.name for path in MODEL_DIR.glob("*.gguf")])
    if not models:
        _maybe_use_cwd_models()
        models = sorted([path.name for path in MODEL_DIR.glob("*.gguf")])
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
    path = Path(candidate)
    if path.is_absolute() and path.exists():
        return path
    fallback = MODEL_DIR / candidate
    if fallback.exists():
        return fallback
    raise HTTPException(status_code=404, detail=f"Model not found: {candidate}")


def _load_model(tipo_model: str, device: str):
    if tipo_model == _current_model["name"] and _current_model["instance"] is not None:
        return _current_model["instance"]

    model_path = _resolve_model_path(tipo_model)
    n_gpu_layers = DEFAULT_GPU_LAYERS if device == "cuda" else 0
    _logger.info("Loading model: %s (device=%s)", model_path, device)
    model = Llama(
        model_path=str(model_path),
        n_ctx=384,
        n_gpu_layers=n_gpu_layers,
        seed=0,
        verbose=False,
    )
    _current_model["name"] = tipo_model
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
    max_new_tokens=256,
    max_retry=5,
):
    prev_len = 0
    retry = max_retry
    llm_gen = ""

    while True:
        try:
            result = model.create_completion(
                prompt,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                max_tokens=max_new_tokens,
                repeat_penalty=1.0,
                seed=seed if seed is not None else 0,
            )
        except TypeError:
            _logger.warning("llama_cpp missing min_p support; retrying without it")
            result = model.create_completion(
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_new_tokens,
                repeat_penalty=1.0,
                seed=seed if seed is not None else 0,
            )
        llm_gen = prompt + result["choices"][0]["text"]
        extra = llm_gen.split("<|input_end|>")[-1].strip().strip(",")
        raw_tokens = [tok.strip() for tok in extra.split(",") if tok.strip()]
        extra_tokens = list({tok for tok in raw_tokens if not _black_list_match(tok, black_list)})
        llm_gen = llm_gen.replace(extra, ", ".join(extra_tokens))

        if len(prompt_tags) + len(extra_tokens) < len_target:
            if len(extra_tokens) == prev_len and prev_len > 0:
                if retry < 0:
                    break
                retry -= 1
            prev_len = len(extra_tokens)
            prompt = llm_gen.strip().replace("  <|", " <|")
        else:
            break

    return llm_gen, extra_tokens


@APP.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _logger.exception("Unhandled error for %s %s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@APP.get("/health")
def health():
    if not LLAMA_AVAILABLE:
        return {"status": "error", "detail": str(LLAMA_IMPORT_ERROR)}
    return {"status": "ok"}


@APP.get("/models")
def list_models():
    _assert_llama_available()
    models = _list_models()
    try:
        entries = [p.name for p in MODEL_DIR.iterdir()]
    except Exception as exc:
        entries = [f"<error:{exc}>\n"]
    _logger.info(
        "Model list requested: %d models (dir=%s entries=%s)",
        len(models),
        MODEL_DIR,
        entries,
    )
    return {"models": models}


@APP.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    _assert_llama_available()

    if req.seed is None or req.seed < 0:
        seed = int.from_bytes(os.urandom(4), "little") % SEED_MAX
    else:
        seed = min(req.seed, SEED_MAX)

    tag_length = req.tag_length.replace(" ", "_")
    len_target = TARGET.get(tag_length, TARGET["long"])

    with _model_lock:
        model = _load_model(req.tipo_model, req.device)

    prompt_parse_strength = _parse_prompt_attention(req.tags)
    strength_map_nl = []
    nl_prompt = req.nl_prompt or ""
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
    dtg_prompt = apply_dtg_prompt(org_tag_map, tag_length, aspect_ratio)

    _llm_gen, extra_tokens = _generate_tags(
        model,
        dtg_prompt,
        org_tag_map.get("special", []) + org_tag_map.get("general", []),
        len_target,
        black_list,
        temperature=req.temperature,
        top_p=req.top_p,
        min_p=req.min_p,
        top_k=req.top_k,
        seed=seed,
    )

    addon_tags = [tag for tag in extra_tokens if tag not in all_tags]
    tag_map = seperate_tags(all_tags + addon_tags)

    formatted_user_prompt = apply_format(
        _apply_strength(_clone_tag_map(org_tag_map), strength_map, strength_map_nl),
        req.format,
    )

    formatted_prompt_by_tipo = apply_format(
        _apply_strength(_clone_tag_map(tag_map), strength_map, strength_map_nl),
        req.format,
    )
    unformatted_prompt_by_user = req.tags + (f"\n{nl_prompt}" if nl_prompt else "")
    unformatted_prompt_by_tipo = req.tags
    if addon_tags:
        unformatted_prompt_by_tipo += ", " + ", ".join(addon_tags)
    if nl_prompt:
        unformatted_prompt_by_tipo += f"\n{nl_prompt}"

    _logger.info(
        "Generated prompt: tags=%d addon=%d seed=%s",
        len(all_tags),
        len(addon_tags),
        seed,
    )

    return GenerateResponse(
        formatted_prompt=formatted_prompt_by_tipo,
        formatted_user_prompt=formatted_user_prompt,
        unformatted_prompt=unformatted_prompt_by_tipo,
        unformatted_user_prompt=unformatted_prompt_by_user,
    )


app = APP
