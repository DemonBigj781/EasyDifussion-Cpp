#!/usr/bin/env python3
"""Regression checks for LTX native routing and allocation failures."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ltx_conditioning_has_no_empty_tensor_assertion_path() -> None:
    source = (ROOT / "source/sdkit3-port-source/stable-diffusion.cpp/src/conditioning/conditioner.hpp").read_text()
    ltx_start = source.index("struct LTXAVEmbedder")
    ltx_source = source[ltx_start:]

    assert "if (hidden_states.empty())" in ltx_source
    assert "GGML_ASSERT(!hidden_states.empty())" not in ltx_source


def test_video_generation_rejects_failed_conditioning() -> None:
    source = (ROOT / "source/sdkit3-port-source/stable-diffusion.cpp/src/stable-diffusion.cpp").read_text()

    assert "if (embeds.cond.c_crossattn.empty())" in source
    assert "Video conditioning failed" in source


def test_video_routing_preserves_explicit_device_selection() -> None:
    paths = (
        ROOT / "source/UI.cpp/Pages/src/Plugin/plugin_scripts/native-device-routing.js",
        ROOT / "ui/plugins/ui/image_plugin/native-device-routing.plugin.js",
    )

    for path in paths:
        source = path.read_text()
        assert "migrateLegacyVideoRouting" not in source
        assert "nvidia3060" not in source
        assert "assignmentFor" in source


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
    print("ltx native failure regression checks passed")
