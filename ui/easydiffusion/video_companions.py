"""Discover companion weights stored beside standalone native video models."""

from pathlib import Path


MODEL_EXTENSIONS = (".safetensors", ".sft", ".gguf", ".ckpt")


def _ranked_model(directory: Path, preferences: tuple[str, ...]):
    if not directory.is_dir():
        return None
    candidates = [
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
    ]
    if not candidates:
        return None

    def rank(path: Path):
        name = path.name.lower()
        preference = next(
            (index for index, token in enumerate(preferences) if token in name),
            len(preferences),
        )
        return preference, name, str(path)

    return str(min(candidates, key=rank))


def discover_mochi_companions(checkpoint_path) -> dict[str, str]:
    """Find the preferred VAE and T5 XXL stored in a Mochi model directory."""
    checkpoint = Path(str(checkpoint_path)).expanduser().resolve()
    model_root = checkpoint.parent
    if model_root.name.lower() in {"transformer", "diffusion_model", "diffusion_models"}:
        model_root = model_root.parent

    vae = _ranked_model(
        model_root / "vae",
        ("fp8_e4m3fn", "fp8", "scaled", "mochi_vae"),
    )
    text_encoder = _ranked_model(
        model_root / "t5xxl",
        ("q4_0", "q4", "fp8_e4m3fn", "fp8", "t5xxl"),
    )
    companions = {}
    if vae:
        companions["vae"] = vae
    if text_encoder:
        companions["text-encoder"] = text_encoder
    return companions
