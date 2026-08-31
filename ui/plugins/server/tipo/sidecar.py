"""Per-model settings for the built-in TIPO and DanTagGen runtimes."""

from copy import deepcopy
from pathlib import Path

from fastapi import HTTPException
from ruamel.yaml import YAML


SCHEMA = "easy-diffusion.tipo-sidecar.v1"

DTG_FORMAT = """<|special|>,
<|characters|>, <|copyrights|>,
<|artist|>,

<|general|>,

<|quality|>, <|meta|>, <|rating|>"""

TIPO_FORMAT = """<|special|>, <|characters|>, <|copyrights|>,
<|artist|>,

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>"""

PROTOCOL_DEFAULTS = {
    "tipo": {
        "context_length": 1024,
        "temperature": 0.5,
        "top_p": 0.95,
        "min_p": 0.05,
        "top_k": 80,
        "repeat_penalty": 1.17,
        "max_new_tokens": 512,
        "max_retry": 10,
        "max_same_output": 5,
        "tag_length": "long",
        "nl_length": "long",
        "format": TIPO_FORMAT,
    },
    "dantaggen": {
        "context_length": 1024,
        "temperature": 1.0,
        "top_p": 0.85,
        "min_p": 0.0,
        "top_k": 70,
        "repeat_penalty": 1.0,
        "max_new_tokens": 512,
        "max_retry": 20,
        "max_same_output": 15,
        "tag_length": "long",
        "nl_length": "long",
        "format": DTG_FORMAT,
    },
}

SUPPORTED_PROTOCOLS = frozenset(PROTOCOL_DEFAULTS)
SUPPORTED_LENGTHS = frozenset(("very_short", "short", "long", "very_long"))
GENERATION_FIELDS = frozenset(
    (
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "repeat_penalty",
        "max_new_tokens",
        "max_retry",
        "max_same_output",
        "tag_length",
        "nl_length",
    )
)


def detect_protocol(model_path: Path) -> str:
    return "dantaggen" if "dantaggen" in model_path.name.lower() else "tipo"


def sidecar_candidates(model_path: Path):
    """Prefer the conventional model.yaml name, while accepting model.gguf.yaml."""
    yield model_path.with_suffix(".yaml")
    yield model_path.with_suffix(".yml")
    yield Path(f"{model_path}.yaml")
    yield Path(f"{model_path}.yml")


def find_sidecar(model_path: Path):
    return next((path for path in sidecar_candidates(model_path) if path.is_file()), None)


def default_sidecar(model_path: Path, context_length=None):
    protocol = detect_protocol(model_path)
    defaults = deepcopy(PROTOCOL_DEFAULTS[protocol])
    if context_length:
        defaults["context_length"] = min(int(context_length), 1024)
    context_length = defaults.pop("context_length")
    prompt_format = defaults.pop("format")
    return {
        "schema": SCHEMA,
        "protocol": protocol,
        "context_length": context_length,
        "generation": defaults,
        "output": {"format": prompt_format},
    }


def _read_yaml(path: Path):
    try:
        if path.stat().st_size > 64 * 1024:
            raise ValueError("sidecar exceeds 64 KiB")
        yaml = YAML(typ="safe")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle)
    except Exception as error:
        raise ValueError(f"Could not read {path.name}: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    return data


def _number(value, name, *, minimum=None, maximum=None, integer=False):
    expected = int if integer else float
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = expected(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def load_model_settings(model_path: Path, *, strict=True):
    sidecar_path = find_sidecar(model_path)
    try:
        raw = _read_yaml(sidecar_path) if sidecar_path else {}
    except ValueError as error:
        detail = f"Invalid TIPO sidecar for {model_path.name}: {error}"
        if strict:
            raise HTTPException(status_code=400, detail=detail) from error
        return {"error": detail, "sidecar": sidecar_path}
    protocol = str(raw.get("protocol") or detect_protocol(model_path)).strip().lower()
    if protocol not in SUPPORTED_PROTOCOLS:
        error = f"Unsupported TIPO sidecar protocol: {protocol}"
        if strict:
            raise HTTPException(status_code=400, detail=error)
        return {"error": error, "sidecar": sidecar_path}

    settings = deepcopy(PROTOCOL_DEFAULTS[protocol])
    try:
        schema = raw.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"unsupported schema: {schema}")

        context_length = raw.get("context_length", settings["context_length"])
        settings["context_length"] = _number(
            context_length, "context_length", minimum=128, maximum=32768, integer=True
        )

        generation = raw.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("generation must be a mapping")
        unknown = set(generation) - GENERATION_FIELDS
        if unknown:
            raise ValueError(f"unknown generation settings: {', '.join(sorted(unknown))}")
        settings.update(generation)

        output = raw.get("output", {})
        if not isinstance(output, dict):
            raise ValueError("output must be a mapping")
        unknown_output = set(output) - {"format"}
        if unknown_output:
            raise ValueError(f"unknown output settings: {', '.join(sorted(unknown_output))}")
        if "format" in output:
            if not isinstance(output["format"], str):
                raise ValueError("output.format must be text")
            settings["format"] = output["format"]

        settings["temperature"] = _number(
            settings["temperature"], "temperature", minimum=0.0, maximum=5.0
        )
        settings["top_p"] = _number(settings["top_p"], "top_p", minimum=0.0, maximum=1.0)
        settings["min_p"] = _number(settings["min_p"], "min_p", minimum=0.0, maximum=1.0)
        settings["top_k"] = _number(settings["top_k"], "top_k", minimum=0, integer=True)
        settings["repeat_penalty"] = _number(
            settings["repeat_penalty"], "repeat_penalty", minimum=0.0, maximum=5.0
        )
        settings["max_new_tokens"] = _number(
            settings["max_new_tokens"], "max_new_tokens", minimum=1, maximum=4096, integer=True
        )
        settings["max_retry"] = _number(
            settings["max_retry"], "max_retry", minimum=0, maximum=100, integer=True
        )
        settings["max_same_output"] = _number(
            settings["max_same_output"], "max_same_output", minimum=0, maximum=100, integer=True
        )
        for field in ("tag_length", "nl_length"):
            settings[field] = str(settings[field]).replace(" ", "_")
            if settings[field] not in SUPPORTED_LENGTHS:
                raise ValueError(f"{field} must be one of {', '.join(sorted(SUPPORTED_LENGTHS))}")
        if not isinstance(settings["format"], str):
            raise ValueError("format must be text")
    except ValueError as error:
        detail = f"Invalid TIPO sidecar for {model_path.name}: {error}"
        if strict:
            raise HTTPException(status_code=400, detail=detail) from error
        return {"error": detail, "sidecar": sidecar_path}

    settings["protocol"] = protocol
    settings["schema"] = SCHEMA
    settings["sidecar"] = sidecar_path
    return settings


def public_settings(settings, model_path: Path):
    data = {key: value for key, value in settings.items() if key != "sidecar"}
    sidecar_path = settings.get("sidecar")
    data["sidecar"] = (
        sidecar_path.relative_to(model_path.parent).as_posix() if sidecar_path else None
    )
    return data
