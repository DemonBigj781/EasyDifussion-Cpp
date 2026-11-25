"""Exercise the legacy Python video request builder without starting a backend."""

from types import SimpleNamespace

from easydiffusion.backends import webui_common
from easydiffusion.utils import model_identifier


class FakeResponse:
    status_code = 200
    text = ""

    @staticmethod
    def json():
        return {"frames": ["frame"], "fps": 8}


def test_animatediff_video_keeps_both_interposer_payloads():
    captured = {}

    def fake_post(path, json):
        captured["path"] = path
        captured["json"] = json
        return FakeResponse()

    webui_common.USE_SDKIT3_API = True
    webui_common.curr_models["stable-diffusion"] = "/models/sd15-checkpoint.safetensors"
    webui_common.resolve_model_to_use = lambda model, model_type: f"/resolved/{model_type}/{model}"
    webui_common.webui_post = fake_post
    webui_common.print_request = lambda *_args, **_kwargs: None
    model_identifier.identify_model_type = lambda _path: "sd15"
    model_identifier.model_type_to_latent_family = lambda _model_type: "v1"
    model_identifier.identify_vae_latent_family = lambda _path: "xl"

    context = SimpleNamespace(
        model_paths={
            "stable-diffusion": "/models/sd15-checkpoint.safetensors",
            "vae": "/models/sdxl-vae.safetensors",
        }
    )
    result = webui_common.generate_video(
        context,
        init_image="aW1hZ2U=",
        motion_module_model="/models/mm_sd_v15_v2.safetensors",
        latent_interposer_enabled=True,
        latent_interposer_source="xl",
        latent_interposer_encode_enabled=True,
        latent_interposer_decode_enabled=True,
    )

    assert result["frames"] == ["frame"]
    assert captured["path"] == "/sdapi/v1/img2video"
    request = captured["json"]
    assert request["motion_module_path"] == "/models/mm_sd_v15_v2.safetensors"
    assert request["latent_interposer"] == {
        "enabled": True,
        "source": "xl",
        "model_path": "/resolved/latent-interposer/xl-to-v1_interposer-v4.0.safetensors",
        "phase_x": 0,
        "phase_y": 0,
    }
    assert request["vae_interposer"] == {
        "enabled": True,
        "vae_family": "xl",
        "model_family": "v1",
        "encode_model_path": "/resolved/latent-interposer/xl-to-v1_interposer-v4.0",
        "decode_model_path": "/resolved/latent-interposer/v1-to-xl_interposer-v4.0",
    }


if __name__ == "__main__":
    test_animatediff_video_keeps_both_interposer_payloads()
    print("AnimateDiff interpose request payload: OK")
