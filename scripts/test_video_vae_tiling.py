#!/usr/bin/env python3
"""Regression checks for native video VAE tiling propagation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/sdkit3-port-source/src/image_generator.cpp"


def _function_source(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


def test_image_and_video_generation_share_vae_tiling_configuration() -> None:
    source = SOURCE.read_text()
    video = _function_source(
        source,
        "std::vector<std::string> ImageGenerator::generateVideo",
        "std::vector<std::string> ImageGenerator::generateInternal",
    )
    image = _function_source(
        source,
        "std::vector<std::string> ImageGenerator::generateInternal",
        "std::vector<sd_image_t> ImageGenerator::createRefImages",
    )

    assert "configureVaeTiling(gen_params.vae_tiling_params, vae_tiling_)" in video
    assert "configureVaeTiling(gen_params.vae_tiling_params, vae_tiling_enabled)" in image


if __name__ == "__main__":
    test_image_and_video_generation_share_vae_tiling_configuration()
    print("video VAE tiling regression checks passed")
