"""Static contract checks for the AnimateDiff motion-module path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def require(relative_path, *needles):
    content = text(relative_path)
    missing = [needle for needle in needles if needle not in content]
    assert not missing, f"{relative_path} is missing: {missing}"


def require_section(relative_path, start, end, *needles):
    content = text(relative_path)
    assert start in content, f"{relative_path} is missing section start: {start}"
    section = content.split(start, 1)[1]
    assert end in section, f"{relative_path} is missing section end: {end}"
    section = section.split(end, 1)[0]
    missing = [needle for needle in needles if needle not in section]
    assert not missing, f"{relative_path} section {start!r} is missing: {missing}"


def test_native_motion_module_request_contract():
    require(
        "source/sdkit3-port-source/include/image_generator.h",
        "std::string motion_module_path;",
        "const std::string& motion_module_path = \"\"",
    )
    require(
        "source/sdkit3-port-source/src/server.cpp",
        'json_body.has("motion_module_path")',
        "params.motion_module_path = std::string(json_body[\"motion_module_path\"].s())",
    )
    require(
        "source/sdkit3-port-source/src/image_generator.cpp",
        "params.motion_module_path",
        "motion_module_path,",
        'motion_module_path.empty() ? nullptr : motion_module_path.c_str()',
    )
    require(
        "source/sdkit3-port-source/src/logging.cpp",
        "sd_get_last_error()",
        "details += native_error",
    )


def test_model_catalog_does_not_load_motion_modules_as_backend_models():
    for relative_path in (
        "ui/easydiffusion/model_manager/__init__.py",
        "ui/plugins/server/Browse_files/model_manager/__init__.py",
    ):
        require(
            relative_path,
            '"motion-module"',
            '"animatediff_models"',
            'if model_type == "motion-module":',
        )
    require(
        "ui/easydiffusion/tasks/render_video.py",
        'model_type="motion-module"',
        "self.request.motion_module_model = model_manager.resolve_model_to_use",
    )


def test_legacy_video_select_clear_and_frame_limit_contract():
    require(
        "ui/easydiffusion/types.py",
        "motion_module_model: Optional[str] = None",
    )
    require(
        "ui/easydiffusion/backends/webui_common.py",
        "motion_module_model: str = None",
        'cmd["motion_module_path"] = motion_module_model',
    )
    require(
        "ui/plugins/ui/video_plugin/native-video.plugin.html",
        'id="native-video-motion-module"',
        "AnimateDiff / SDXL beta",
    )
    require(
        "ui/plugins/ui/video_plugin/native-video.plugin.js",
        '"motion-module"',
        "if (!event.reqBody.video_mode) return false",
        "event.reqBody.motion_module_model ??=",
        "1, 32, 16",
    )


def test_cpp_video_switch_and_diagnostic_contract():
    require(
        "source/UI.cpp/Pages/src/Generate/options.cpp",
        'id", "motion_module_model"',
        "AnimateDiff frames",
    )
    require(
        "source/UI.cpp/Pages/src/Plugin/plugin_scripts/generate.js",
        'request.motion_module_model = motionModulePicker.value',
        'fetch(isVideo ? "/video" : "/render"',
        "payload.detail",
    )


def test_cache_context_requires_active_multiframe_animatediff():
    require(
        "source/sdkit3-port-source/stable-diffusion.cpp/src/stable-diffusion.cpp",
        "animatediff_loaded && animatediff_num_frames > 1",
        "img_gen_params.cache             = sd_vid_gen_params->cache;",
    )


def test_animatediff_attention_uses_enabled_flash_pipeline():
    require(
        "source/sdkit3-port-source/stable-diffusion.cpp/src/model/diffusion/animatediff.hpp",
        "ctx->flash_attn_enabled",
    )
    require(
        "source/sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/fattn.cu",
        "case 160:",
        "ggml_cuda_xformers_attn_supported(device, dst)",
    )


def test_video_interposer_request_contract():
    require_section(
        "source/sdkit3-port-source/include/image_generator.h",
        "struct VideoGenerationParams {",
        "};",
        "latent_interposer_model_path",
        "latent_interposer_enabled",
        "latent_interposer_source",
        "latent_interposer_source_image_base64",
        "furception_vae_path",
        "latent_interposer_encode_model_path",
        "latent_interposer_decode_model_path",
        "latent_interposer_vae_format",
    )
    require(
        "source/sdkit3-port-source/src/server.cpp",
        "parse_latent_interposer_options(json_body, params)",
        "parse_vae_interposer_options(json_body, params)",
    )
    require(
        "source/sdkit3-port-source/src/image_generator.cpp",
        "params.latent_interposer_model_path",
        "params.furception_vae_path",
        "params.latent_interposer_encode_model_path",
        "params.latent_interposer_decode_model_path",
        "gen_params.latent_interposer_enabled",
        "gen_params.latent_interposer_source_image",
        "freeImage(latent_interposer_source_image)",
    )
    require_section(
        "source/sdkit3-port-source/stable-diffusion.cpp/include/stable-diffusion.h",
        "} sd_img_gen_params_t;",
        "} sd_vid_gen_params_t;",
        "latent_interposer_enabled",
        "latent_interposer_source",
        "latent_interposer_source_image",
        "latent_interposer_phase_x",
        "latent_interposer_phase_y",
    )
    require(
        "source/sdkit3-port-source/stable-diffusion.cpp/src/stable-diffusion.cpp",
        "img_gen_params.latent_interposer_enabled",
        "img_gen_params.latent_interposer_source",
        "img_gen_params.latent_interposer_source_image",
        "img_gen_params.latent_interposer_phase_x",
        "img_gen_params.latent_interposer_phase_y",
    )
    require_section(
        "ui/easydiffusion/types.py",
        "class VideoGenerationRequest(BaseModel):",
        "class ModelsData(BaseModel):",
        "latent_interposer_enabled",
        "latent_interposer_source",
        "latent_interposer_encode_enabled",
        "latent_interposer_decode_enabled",
    )
    require_section(
        "ui/easydiffusion/backends/webui_common.py",
        "def generate_video(",
        "def filter_images(",
        "latent_interposer_enabled: bool = False",
        "latent_interposer_encode_enabled: bool = False",
        "latent_interposer_decode_enabled: bool = False",
        "_add_latent_interposer_options(",
        "_add_vae_interposer_options(",
    )


if __name__ == "__main__":
    test_native_motion_module_request_contract()
    test_model_catalog_does_not_load_motion_modules_as_backend_models()
    test_legacy_video_select_clear_and_frame_limit_contract()
    test_cpp_video_switch_and_diagnostic_contract()
    test_cache_context_requires_active_multiframe_animatediff()
    test_animatediff_attention_uses_enabled_flash_pipeline()
    test_video_interposer_request_contract()
    print("AnimateDiff motion-module contract: OK")
