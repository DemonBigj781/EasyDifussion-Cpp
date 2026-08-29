import os
import requests
from requests.exceptions import ConnectTimeout, ConnectionError, ReadTimeout
from typing import Union, List
from threading import local as Context
from threading import local
from threading import Thread
import uuid
import time
from copy import deepcopy
import atexit
import numpy as np
from PIL import Image

from sdkit.utils import base64_str_to_img, img_to_base64_str, log
from torchruntime.utils import get_device

from easydiffusion.app import getConfig
from easydiffusion.model_manager import get_model_dirs, resolve_model_to_use

from common import kill

WEBUI_HOST = "localhost"
WEBUI_PORT = "7860"
WEBUI_API_PREFIX = ""
USE_SDKIT3_API = False

DEFAULT_WEBUI_OPTIONS = {
    "show_progress_every_n_steps": 3,
    "show_progress_grid": True,
    "live_previews_enable": False,
    "forge_additional_modules": [],
}

MODELS_TO_OVERRIDE = {
    "stable-diffusion": "--ckpt-dir",
    "vae": "--vae-dir",
    "hypernetwork": "--hypernetwork-dir",
    "gfpgan": "--gfpgan-models-path",
    "realesrgan": "--realesrgan-models-path",
    "lora": "--lora-dir",
    "codeformer": "--codeformer-models-path",
    "embeddings": "--embeddings-dir",
    "controlnet": "--controlnet-dir",
    "text-encoder": "--text-encoder-dir",
}

backend_process = None

webui_opts: dict = None


curr_models = {
    "stable-diffusion": None,
    "vae": None,
    "text-encoder": None,
}


def set_options(context, **kwargs):
    changed_opts = {}

    opts_mapping = {
        "stream_image_progress": ("live_previews_enable", bool),
        "stream_image_progress_interval": ("show_progress_every_n_steps", int),
        "clip_skip": ("CLIP_stop_at_last_layers", int),
        "clip_skip_sdxl": ("sdxl_clip_l_skip", bool),
        "vae_tiling": ("vae_tiling", bool),
        "output_format": ("samples_format", str),
    }

    for ed_key, webui_key in opts_mapping.items():
        webui_key, webui_type = webui_key

        if ed_key in kwargs and (
            webui_opts is None
            or webui_key not in webui_opts
            or webui_opts[webui_key] != webui_type(kwargs[ed_key])
        ):
            changed_opts[webui_key] = webui_type(kwargs[ed_key])

    if changed_opts:
        changed_opts["sd_model_checkpoint"] = curr_models["stable-diffusion"]

        print(f"Got options: {kwargs}. Sending options: {changed_opts}")

        try:
            res = webui_post("/sdapi/v1/options", json=changed_opts)
            if res.status_code != 200:
                raise Exception(res.text)

            webui_opts.update(changed_opts)
        except Exception as e:
            print(f"Error setting options: {e}")


def ping(timeout=1):
    "timeout (in seconds)"

    global webui_opts

    try:
        res = webui_get("/internal/ping", timeout=timeout)

        if res.status_code != 200:
            raise ConnectTimeout(res.text)

        if webui_opts is None:
            try:
                res = webui_post("/sdapi/v1/options", json=DEFAULT_WEBUI_OPTIONS)
                if res.status_code != 200:
                    raise Exception(res.text)
            except Exception as e:
                print(f"Error setting options: {e}")

            try:
                res = webui_get("/sdapi/v1/options")
                if res.status_code != 200:
                    raise Exception(res.text)

                webui_opts = res.json()
            except Exception as e:
                print(f"Error getting options: {e}")

        return True
    except (ConnectTimeout, ConnectionError, ReadTimeout) as e:
        raise TimeoutError(e)


def load_model(context, model_type, **kwargs):
    from easydiffusion.app import ROOT_DIR, getConfig

    config = getConfig()
    models_dir = config.get("models_dir", os.path.join(ROOT_DIR, "models"))

    model_path = context.model_paths[model_type]

    if model_type == "stable-diffusion":
        base_dir = get_model_dirs(model_type, models_dir)[0]
        model_path = os.path.relpath(model_path, base_dir)

    # print(f"load model: {model_type=} {model_path=} {curr_models=}")
    curr_models[model_type] = model_path


def unload_model(context, model_type, **kwargs):
    # print(f"unload model: {model_type=} {curr_models=}")
    curr_models[model_type] = None


def flush_model_changes(context):
    if webui_opts is None:
        print("Server not ready, can't set the model")
        return

    modules = []
    for model_type in ("vae", "text-encoder"):
        if curr_models[model_type]:
            model_paths = curr_models[model_type]
            model_paths = [model_paths] if not isinstance(model_paths, list) else model_paths
            modules += model_paths

    opts = {"sd_model_checkpoint": curr_models["stable-diffusion"], "forge_additional_modules": modules}

    print("Setting backend models", opts)

    try:
        res = webui_post("/sdapi/v1/options", json=opts)
        print("got res", res.status_code)
        if res.status_code != 200:
            raise Exception(res.text)
    except Exception as e:
        raise RuntimeError(
            f"The engine failed to set the required options. Please check the logs in the command line window for more details."
        )


def generate_images(
    context: Context,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    width: int = 512,
    height: int = 512,
    num_outputs: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    distilled_guidance_scale: float = 3.5,
    init_image=None,
    init_image_mask=None,
    ref_images=None,
    control_image=None,
    control_alpha=1.0,
    controlnet_filter=None,
    controlnet_union_type: str = "canny",
    control_net_lllite_image=None,
    control_net_lllite_model=None,
    control_net_lllite_strength: float = 1.0,
    control_net_lllite_start_percent: float = 0.0,
    control_net_lllite_end_percent: float = 100.0,
    ip_adapter_image=None,
    ip_adapter_model=None,
    ip_adapter_clip_vision=None,
    ip_adapter_strength: float = 1.0,
    ip_adapter_start_percent: float = 0.0,
    ip_adapter_end_percent: float = 100.0,
    latent_interposer_model=None,
    latent_interposer_enabled: bool = False,
    latent_interposer_source: str = "fx",
    latent_interposer_furception_vae: str = "furception_vae_1-0",
    latent_interposer_source_seed: int = 1,
    latent_interposer_phase_x: int = 0,
    latent_interposer_phase_y: int = 0,
    latent_interposer_encode_enabled: bool = False,
    latent_interposer_decode_enabled: bool = False,
    latent_interposer_encode_model=None,
    latent_interposer_decode_model=None,
    prompt_strength: float = 0.8,
    preserve_init_image_color_profile=False,
    strict_mask_border=False,
    sampler_name: str = "euler_a",
    scheduler_name: str = "simple",
    hypernetwork_strength: float = 0,
    tiling=None,
    lora_alpha: Union[float, List[float]] = 0,
    sampler_params={},
    callback=None,
    output_type="pil",
):

    task_id = str(uuid.uuid4())

    sampler_name = convert_ED_sampler_names(sampler_name)
    controlnet_filter = convert_ED_controlnet_filter_name(controlnet_filter)

    cmd = {
        "force_task_id": task_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler_name": sampler_name,
        "scheduler": scheduler_name,
        "steps": num_inference_steps,
        "seed": seed,
        "cfg_scale": guidance_scale,
        "distilled_cfg_scale": distilled_guidance_scale,
        "batch_size": num_outputs,
        "width": width,
        "height": height,
    }

    if init_image:
        cmd["init_images"] = [init_image]
        cmd["denoising_strength"] = prompt_strength
    if init_image_mask:
        cmd["mask"] = init_image_mask if isinstance(init_image_mask, str) else img_to_base64_str(init_image_mask)
        cmd["include_init_images"] = True
        cmd["inpainting_fill"] = 1
        cmd["initial_noise_multiplier"] = 1
        cmd["inpaint_full_res"] = 0
        cmd["inpaint_full_res_padding"] = 32
        cmd["resize_mode"] = 1
        cmd["mask_blur"] = 4
    if ref_images:
        cmd["ref_images"] = ref_images if isinstance(ref_images, list) else [ref_images]

    if control_net_lllite_image and control_net_lllite_model:
        if not USE_SDKIT3_API:
            raise RuntimeError("ControlNet-LLLite requires the sdkit3 backend")
        lllite_path = resolve_model_to_use(control_net_lllite_model, "controlnet-lllite")
        cmd["controlnet_lllite"] = {
            "model_path": lllite_path,
            "image": control_net_lllite_image,
            "strength": float(control_net_lllite_strength),
            "start_percent": float(control_net_lllite_start_percent),
            "end_percent": float(control_net_lllite_end_percent),
        }

    if ip_adapter_image or ip_adapter_model or ip_adapter_clip_vision:
        if not USE_SDKIT3_API:
            raise RuntimeError("Native IP-Adapter requires the sdkit3 backend")
        if not ip_adapter_image or not ip_adapter_model or not ip_adapter_clip_vision:
            raise ValueError("IP-Adapter requires an image, adapter model, and CLIP Vision model")

        raw_start = max(0.0, min(100.0, float(ip_adapter_start_percent)))
        raw_end = max(0.0, min(100.0, float(ip_adapter_end_percent)))
        cmd["ip_adapter"] = {
            "model_path": resolve_model_to_use(ip_adapter_model, "ip-adapter"),
            "clip_vision_path": resolve_model_to_use(ip_adapter_clip_vision, "clip-vision"),
            "image": ip_adapter_image,
            "strength": float(ip_adapter_strength),
            "start_percent": min(raw_start, raw_end),
            "end_percent": max(raw_start, raw_end),
        }

    if latent_interposer_enabled:
        if not USE_SDKIT3_API:
            raise RuntimeError("Latent Interposer requires the sdkit3 backend")
        source = str(latent_interposer_source).lower()
        if source not in ("v1", "xl", "v3", "fx"):
            raise ValueError("Latent Interposer source must be v1, xl, v3, or fx")

        checkpoint = str(curr_models.get("stable-diffusion") or "").lower()
        if "flux" in checkpoint:
            target = "fx"
        elif "sd3" in checkpoint or "stable-diffusion-3" in checkpoint:
            target = "v3"
        elif "sdxl" in checkpoint or "/xl" in checkpoint or checkpoint.startswith("xl"):
            target = "xl"
        else:
            target = "v1"

        if source != target and target not in ("v1", "xl", "v3"):
            raise ValueError(f"city96 v4.0 has no {source}-to-{target} converter")

        interposer_path = ""
        if source != target:
            expected_model = f"{source}-to-{target}_interposer-v4.0.safetensors"
            selected_model = latent_interposer_model or expected_model
            if not os.path.basename(str(selected_model)).startswith(f"{source}-to-{target}_interposer-v4.0"):
                raise ValueError(
                    f"Selected converter does not match the required {source}-to-{target} conversion"
                )
            interposer_path = resolve_model_to_use(selected_model, "latent-interposer")

        cmd["latent_interposer"] = {
            "enabled": True,
            "source": source,
            "model_path": interposer_path,
            "phase_x": int(latent_interposer_phase_x),
            "phase_y": int(latent_interposer_phase_y),
        }
        if source == "v1":
            furception_path = resolve_model_to_use(latent_interposer_furception_vae, "furception-vae")
            source_seed = max(0, int(latent_interposer_source_seed))
            rng = np.random.default_rng(source_seed)
            pixels = rng.integers(0, 256, (int(height), int(width), 3), dtype=np.uint8)
            cmd["latent_interposer"]["furception_vae_path"] = furception_path
            cmd["latent_interposer"]["source_image"] = img_to_base64_str(Image.fromarray(pixels, "RGB"))

    if latent_interposer_encode_enabled or latent_interposer_decode_enabled:
        if not USE_SDKIT3_API:
            raise RuntimeError("VAE Encode/Decode Interpose requires the sdkit3 backend")
        from easydiffusion.utils.model_identifier import (
            identify_model_type,
            identify_vae_latent_family,
            model_type_to_latent_family,
        )

        checkpoint_path = context.model_paths.get("stable-diffusion")
        if not checkpoint_path:
            raise ValueError("Cannot detect the destination latent standard without a checkpoint")
        model_family = model_type_to_latent_family(identify_model_type(checkpoint_path))
        if model_family not in ("v1", "xl", "v3", "fx"):
            raise ValueError("The selected checkpoint does not use a supported city96 latent standard")

        vae_path = context.model_paths.get("vae")
        vae_family = identify_vae_latent_family(vae_path) if vae_path else model_family
        if vae_family not in ("v1", "xl", "v3", "fx"):
            raise ValueError("The selected VAE does not use a supported city96 latent standard")
        if vae_family == model_family:
            raise ValueError("Encode/Decode Interpose is unnecessary when the VAE and checkpoint standards match")

        available_conversions = {
            ("v1", "xl"), ("v1", "v3"),
            ("xl", "v1"), ("xl", "v3"),
            ("v3", "v1"), ("v3", "xl"),
            ("fx", "v1"), ("fx", "xl"), ("fx", "v3"),
        }

        def resolve_interposer(enabled, source, destination, selected, stage):
            if not enabled:
                return ""
            if (source, destination) not in available_conversions:
                raise ValueError(f"city96 v4.0 has no {source}-to-{destination} {stage} converter")
            expected = f"{source}-to-{destination}_interposer-v4.0"
            selected = str(selected or expected)
            if not os.path.basename(selected).startswith(expected):
                raise ValueError(
                    f"Selected {stage} converter does not match the required {source}-to-{destination} direction"
                )
            return resolve_model_to_use(selected, "latent-interposer")

        # Encode takes selected-VAE latents into the checkpoint's standard.
        encode_path = resolve_interposer(
            latent_interposer_encode_enabled,
            vae_family,
            model_family,
            latent_interposer_encode_model,
            "encode",
        )
        # Decode is intentionally flipped: checkpoint latents return to the VAE standard.
        decode_path = resolve_interposer(
            latent_interposer_decode_enabled,
            model_family,
            vae_family,
            latent_interposer_decode_model,
            "decode",
        )
        cmd["vae_interposer"] = {
            "enabled": True,
            "vae_family": vae_family,
            "model_family": model_family,
            "encode_model_path": encode_path,
            "decode_model_path": decode_path,
        }

    if context.model_paths.get("lora"):
        lora_model = context.model_paths["lora"]
        lora_model = lora_model if isinstance(lora_model, list) else [lora_model]
        lora_alpha = lora_alpha if isinstance(lora_alpha, list) else [lora_alpha]

        if len(lora_model) != len(lora_alpha):
            raise ValueError("lora_model and lora_alpha must have the same length")

        if USE_SDKIT3_API:
            cmd["lora_paths"] = lora_model
            cmd["lora_alphas"] = lora_alpha
        else:
            for lora, alpha in zip(lora_model, lora_alpha):
                lora = os.path.basename(lora)
                lora = os.path.splitext(lora)[0]
                cmd["prompt"] += f" <lora:{lora}:{alpha}>"

    if control_image and context.model_paths.get("controlnet"):
        controlnet_model = context.model_paths["controlnet"]

        model_hash = auto1111_hash(controlnet_model)
        controlnet_model = os.path.basename(controlnet_model)
        controlnet_model = os.path.splitext(controlnet_model)[0]
        print(f"setting controlnet model: {controlnet_model}")
        controlnet_model = f"{controlnet_model} [{model_hash}]"

        cmd["alwayson_scripts"] = {
            "controlnet": {
                "args": [
                    {
                        "image": control_image,
                        "weight": control_alpha,
                        "module": controlnet_filter or "none",
                        "model": controlnet_model,
                        "union_control_type": controlnet_union_type or "canny",
                        "resize_mode": "Crop and Resize",
                        "threshold_a": 50,
                        "threshold_b": 130,
                    }
                ]
            }
        }

    operation_to_apply = "img2img" if init_image else "txt2img"

    stream_image_progress = webui_opts.get("live_previews_enable", False)

    progress_thread = Thread(
        target=image_progress_thread, args=(task_id, callback, stream_image_progress, num_outputs, num_inference_steps)
    )
    progress_thread.start()

    print(f"task id: {task_id}")
    print_request(operation_to_apply, cmd)

    res = webui_post(f"/sdapi/v1/{operation_to_apply}", json=cmd)
    if res.status_code == 200:
        res = res.json()
    else:
        if res.status_code == 500:
            res = res.json()
            log.error(f"Server error: {res}")
            raise Exception(f"{res['message']}. Please check the logs in the command-line window for more details.")

        raise Exception(
            f"HTTP Status {res.status_code}. The engine failed while generating this image. Please check the logs in the command-line window for more details."
        )

    import json

    print(json.loads(res["info"])["infotexts"])

    images = res["images"]
    if output_type == "pil":
        images = [base64_str_to_img(img) for img in images]
    elif output_type == "base64":
        images = [base64_buffer_to_base64_img(img) for img in images]

    return images


def generate_video(
    context: Context,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    init_image=None,
    end_image=None,
    prompt_strength: float = 0.75,
    video_frames: int = 25,
    fps: int = 8,
    sampler_name: str = "euler",
    scheduler_name: str = "simple",
    flow_shift=None,
    moe_boundary: float = 0.875,
    cache_mode: str = "disabled",
    cache_threshold=None,
    cache_start_percent: float = 15.0,
    cache_end_percent: float = 95.0,
    callback=None,
):
    if not USE_SDKIT3_API:
        raise RuntimeError("Native video generation requires the sdkit3 backend")

    task_id = str(uuid.uuid4())
    cmd = {
        "force_task_id": task_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "steps": int(num_inference_steps),
        "cfg_scale": float(guidance_scale),
        "video_frames": int(video_frames),
        "fps": int(fps),
        "sampler_name": convert_ED_sampler_names(sampler_name),
        "scheduler": scheduler_name,
        "denoising_strength": float(prompt_strength),
        "moe_boundary": float(moe_boundary),
        "cache": {
            "mode": str(cache_mode),
            "start_percent": float(cache_start_percent),
            "end_percent": float(cache_end_percent),
        },
    }
    if flow_shift is not None:
        cmd["flow_shift"] = float(flow_shift)
    if cache_threshold is not None:
        cmd["cache"]["threshold"] = float(cache_threshold)
    if init_image:
        cmd["init_images"] = [init_image]
    if end_image:
        cmd["end_image"] = end_image

    operation = "img2video" if init_image else "txt2video"
    progress_thread = None
    if callback is not None:
        progress_thread = Thread(
            target=image_progress_thread,
            args=(task_id, callback, False, 1, num_inference_steps),
            daemon=True,
        )
        progress_thread.start()

    print_request(operation, cmd)
    res = webui_post(f"/sdapi/v1/{operation}", json=cmd)
    if res.status_code != 200:
        try:
            message = res.json().get("message", res.text)
        except Exception:
            message = res.text
        raise RuntimeError(f"Native video backend failed: {message}")
    result = res.json()
    if progress_thread is not None:
        progress_thread.join(timeout=2)
    return result


def filter_images(context: Context, images, filters, filter_params={}, input_type="pil"):
    """
    * context: Context
    * images: str or PIL.Image or list of str/PIL.Image - image to filter. if a string is passed, it needs to be a base64-encoded image
    * filters: filter_type (string) or list of strings
    * filter_params: dict

    returns: [PIL.Image] - list of filtered images
    """
    images = images if isinstance(images, list) else [images]
    filters = filters if isinstance(filters, list) else [filters]

    if "nsfw_checker" in filters:
        filters.remove("nsfw_checker")  # handled by ED directly

    # Face filters also use the extras API, so start from a no-upscale payload.
    args = {"upscaling_resize": 1, "upscaler_1": "None"}
    controlnet_filters = []

    print(filter_params)

    for filter_name in filters:
        params = filter_params.get(filter_name, {})

        if filter_name == "gfpgan":
            args["gfpgan_visibility"] = 1

        if filter_name in ("realesrgan", "esrgan_4x", "lanczos", "nearest", "scunet", "swinir"):
            args["upscaler_1"] = params.get("upscaler", "RealESRGAN_x4plus")
            args["upscaling_resize"] = params.get("scale", 4)

            if args["upscaler_1"] == "RealESRGAN_x4plus":
                args["upscaler_1"] = "R-ESRGAN 4x+"
            elif args["upscaler_1"] == "RealESRGAN_x4plus_anime_6B":
                args["upscaler_1"] = "R-ESRGAN 4x+ Anime6B"

        if filter_name == "codeformer":
            args["codeformer_visibility"] = 1
            args["codeformer_weight"] = params.get("codeformer_fidelity", 0.5)

        if filter_name.startswith("controlnet_"):
            filter_name = convert_ED_controlnet_filter_name(filter_name)
            controlnet_filters.append(filter_name)

    print(f"filtering {len(images)} images with {args}. {controlnet_filters=}")

    if len(filters) > len(controlnet_filters):
        filtered_images = extra_batch_images(images, input_type=input_type, **args)
    else:
        filtered_images = images

    for filter_name in controlnet_filters:
        filtered_images = controlnet_filter(filtered_images, module=filter_name, input_type=input_type)

    return filtered_images


def get_url():
    return f"//{WEBUI_HOST}:{WEBUI_PORT}/?__theme=dark"


def stop_rendering(context):
    try:
        res = webui_post("/sdapi/v1/interrupt")
        if res.status_code != 200:
            raise Exception(res.text)
    except Exception as e:
        print(f"Error interrupting webui: {e}")


def refresh_models():
    def make_refresh_call(type):
        try:
            webui_post(f"/sdapi/v1/refresh-{type}")
        except:
            pass

    try:
        for type in ("checkpoints", "vae-and-text-encoders"):
            t = Thread(target=make_refresh_call, args=(type,))
            t.start()
    except Exception as e:
        print(f"Error refreshing models: {e}")


def list_controlnet_filters():
    return [
        "openpose",
        "openpose_face",
        "openpose_faceonly",
        "openpose_hand",
        "openpose_full",
        "animal_openpose",
        "densepose_parula (black bg & blue torso)",
        "densepose (pruple bg & purple torso)",
        "dw_openpose_full",
        "mediapipe_face",
        "instant_id_face_keypoints",
        "InsightFace+CLIP-H (IPAdapter)",
        "InsightFace (InstantID)",
        "canny",
        "mlsd",
        "scribble_hed",
        "scribble_hedsafe",
        "scribble_pidinet",
        "scribble_pidsafe",
        "scribble_xdog",
        "softedge_hed",
        "softedge_hedsafe",
        "softedge_pidinet",
        "softedge_pidsafe",
        "softedge_teed",
        "normal_bae",
        "depth_midas",
        "normal_midas",
        "depth_zoe",
        "depth_leres",
        "depth_leres++",
        "depth_anything_v2",
        "depth_anything",
        "depth_hand_refiner",
        "depth_marigold",
        "lineart_coarse",
        "lineart_realistic",
        "lineart_anime",
        "lineart_standard (from white bg & black line)",
        "lineart_anime_denoise",
        "reference_adain",
        "reference_only",
        "reference_adain+attn",
        "tile_colorfix",
        "tile_resample",
        "tile_colorfix+sharp",
        "CLIP-ViT-H (IPAdapter)",
        "CLIP-G (Revision)",
        "CLIP-G (Revision ignore prompt)",
        "CLIP-ViT-bigG (IPAdapter)",
        "InsightFace+CLIP-H (IPAdapter)",
        "inpaint_only",
        "inpaint_only+lama",
        "inpaint_global_harmonious",
        "seg_ufade20k",
        "seg_ofade20k",
        "seg_anime_face",
        "seg_ofcoco",
        "shuffle",
        "segment",
        "invert (from white bg & black line)",
        "threshold",
        "t2ia_sketch_pidi",
        "t2ia_color_grid",
        "recolor_intensity",
        "recolor_luminance",
        "blur_gaussian",
    ]


def controlnet_filter(images, module="none", processor_res=512, threshold_a=64, threshold_b=64, input_type="pil"):
    if input_type == "pil":
        images = [img_to_base64_str(x) for x in images]

    payload = {
        "controlnet_module": module,
        "controlnet_input_images": images,
        "controlnet_processor_res": processor_res,
        "controlnet_threshold_a": threshold_a,
        "controlnet_threshold_b": threshold_b,
    }
    res = webui_post("/controlnet/detect", json=payload)
    res = res.json()
    filtered_images = res["images"]

    if input_type == "pil":
        filtered_images = [base64_str_to_img(img) for img in filtered_images]
    elif input_type == "base64":
        filtered_images = [base64_buffer_to_base64_img(img) for img in filtered_images]

    return filtered_images


def image_progress_thread(task_id, callback, stream_image_progress, total_images, total_steps):
    from PIL import Image

    last_preview_id = -1
    last_step = -1
    reported_total_steps = max(1, int(total_steps))

    EMPTY_IMAGE = Image.new("RGB", (1, 1))

    while True:
        res = webui_post(
            f"/internal/progress",
            json={"id_task": task_id, "live_preview": stream_image_progress, "id_live_preview": last_preview_id},
        )
        if res.status_code == 200:
            res = res.json()
        elif res.status_code == 404:
            time.sleep(0.5)
            continue
        else:
            raise RuntimeError(f"Unexpected progress response. Status code: {res.status_code}. Res: {res.text}")

        last_preview_id = res["id_live_preview"]

        if res["progress"] is not None:
            backend_total_steps = int(res.get("total_steps") or 0)
            if backend_total_steps > 0:
                reported_total_steps = backend_total_steps
            backend_step = res.get("current_step")
            if backend_step is None:
                step_num = int(round(res["progress"] * reported_total_steps))
            else:
                step_num = int(backend_step)
            step_num = max(0, min(reported_total_steps, step_num))

            if res["live_preview"]:
                img = res["live_preview"]
                img = base64_str_to_img(img)
                images = [EMPTY_IMAGE] * total_images
                images[0] = img
            else:
                images = None

            # The progress endpoint is polled more often than many samplers
            # advance. Do not flood Easy Diffusion with duplicate JSON events,
            # but keep a new live preview even when the step did not change.
            if step_num != last_step or images is not None:
                callback(images, step_num)
                last_step = step_num

        if res["completed"] == True:
            if last_step < reported_total_steps:
                callback(None, reported_total_steps)
            print("Complete!")
            break

        time.sleep(0.5)


def webui_get(uri, *args, **kwargs):
    url = f"http://{WEBUI_HOST}:{WEBUI_PORT}{WEBUI_API_PREFIX}{uri}"
    return requests.get(url, *args, **kwargs)


def webui_post(uri, *args, **kwargs):
    url = f"http://{WEBUI_HOST}:{WEBUI_PORT}{WEBUI_API_PREFIX}{uri}"
    return requests.post(url, *args, **kwargs)


def print_request(operation_to_apply, args):
    args = deepcopy(args)
    if "init_images" in args:
        args["init_images"] = ["img" for _ in args["init_images"]]
    if "mask" in args:
        args["mask"] = "mask_img"

    controlnet_args = args.get("alwayson_scripts", {}).get("controlnet", {}).get("args", [])
    if controlnet_args:
        controlnet_args[0]["image"] = "control_image"

    print(f"operation: {operation_to_apply}, args: {args}")


def auto1111_hash(file_path):
    import hashlib

    with open(file_path, "rb") as f:
        f.seek(0x100000)
        b = f.read(0x10000)
        return hashlib.sha256(b).hexdigest()[:8]


def extra_batch_images(
    images,  # list of PIL images
    name_list=None,  # list of image names
    resize_mode=0,
    show_extras_results=True,
    gfpgan_visibility=0,
    codeformer_visibility=0,
    codeformer_weight=0,
    upscaling_resize=2,
    upscaling_resize_w=512,
    upscaling_resize_h=512,
    upscaling_crop=True,
    upscaler_1="None",
    upscaler_2="None",
    extras_upscaler_2_visibility=0,
    upscale_first=False,
    use_async=False,
    input_type="pil",
):
    if name_list is not None:
        if len(name_list) != len(images):
            raise RuntimeError("len(images) != len(name_list)")
    else:
        name_list = [f"image{i + 1:05}" for i in range(len(images))]

    if input_type == "pil":
        images = [img_to_base64_str(x) for x in images]

    image_list = []
    for name, image in zip(name_list, images):
        image_list.append({"data": image, "name": name})

    payload = {
        "resize_mode": resize_mode,
        "show_extras_results": show_extras_results,
        "gfpgan_visibility": gfpgan_visibility,
        "codeformer_visibility": codeformer_visibility,
        "codeformer_weight": codeformer_weight,
        "upscaling_resize": upscaling_resize,
        "upscaling_resize_w": upscaling_resize_w,
        "upscaling_resize_h": upscaling_resize_h,
        "upscaling_crop": upscaling_crop,
        "upscaler_1": upscaler_1,
        "upscaler_2": upscaler_2,
        "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
        "upscale_first": upscale_first,
        "imageList": image_list,
    }

    res = webui_post("/sdapi/v1/extra-batch-images", json=payload)
    if res.status_code == 200:
        res = res.json()
    else:
        raise Exception(
            "The engine failed while filtering this image. Please check the logs in the command-line window for more details."
        )

    images = res["images"]

    if input_type == "pil":
        images = [base64_str_to_img(img) for img in images]
    elif input_type == "base64":
        images = [base64_buffer_to_base64_img(img) for img in images]

    return images


def base64_buffer_to_base64_img(img):
    output_format = webui_opts.get("samples_format", "jpeg")
    mime_type = f"image/{output_format.lower()}"
    return f"data:{mime_type};base64," + img


def convert_ED_sampler_names(sampler_name):
    name_mapping = {
        "dpmpp_2m": "DPM++ 2M",
        "dpmpp_2m_v2": "DPM++ 2M v2",
        "dpmpp_sde": "DPM++ SDE",
        "dpmpp_2m_sde": "DPM++ 2M SDE",
        "dpmpp_2m_sde_heun": "DPM++ 2M SDE Heun",
        "dpmpp_2s_a": "DPM++ 2S a",
        "dpmpp_3m_sde": "DPM++ 3M SDE",
        "euler_a": "Euler a",
        "euler": "Euler",
        "lms": "LMS",
        "heun": "Heun",
        "dpm2": "DPM2",
        "dpm2_a": "DPM2 a",
        "dpm_fast": "DPM fast",
        "dpm_adaptive": "DPM adaptive",
        "restart": "Restart",
        "heun_pp2": "HeunPP2",
        "ipndm": "IPNDM",
        "ipndm_v": "IPNDM_V",
        "deis": "DEIS",
        "ddim": "DDIM",
        "ddim_cfgpp": "DDIM CFG++",
        "plms": "PLMS",
        "unipc": "UniPC",
        "lcm": "LCM",
        "ddpm": "DDPM",
        "forge_flux_realistic": "[Forge] Flux Realistic",
        "forge_flux_realistic_slow": "[Forge] Flux Realistic (Slow)",
        "tcd": "TCD",
        # deprecated samplers in 3.5
        "dpm_solver_stability": None,
        "unipc_snr": None,
        "unipc_tu": None,
        "unipc_snr_2": None,
        "unipc_tu_2": None,
        "unipc_tq": None,
    }
    return name_mapping.get(sampler_name)


def convert_ED_controlnet_filter_name(filter):
    if filter is None:
        return None

    def cn(n):
        if n.startswith("controlnet_"):
            return n[len("controlnet_") :]
        return n

    mapping = {
        "controlnet_scribble_hedsafe": None,
        "controlnet_scribble_pidsafe": None,
        "controlnet_softedge_pidsafe": "controlnet_softedge_pidisafe",
        "controlnet_normal_bae": "controlnet_normalbae",
        "controlnet_segment": None,
    }
    if isinstance(filter, list):
        return [cn(mapping.get(f, f)) for f in filter]
    return cn(mapping.get(filter, filter))


def get_model_path_args(return_string=True):
    args = []
    for model_type, flag in MODELS_TO_OVERRIDE.items():
        model_dir = get_model_dirs(model_type)[0]
        if return_string:
            args.append(f'{flag} "{model_dir}"')
        else:
            args.append(flag)
            args.append(model_dir)

    if return_string:
        return " ".join(args)

    return args


def get_common_cli_args(return_string=True):
    model_path_args = get_model_path_args(return_string=return_string)
    extra_args = ["--port", str(WEBUI_PORT), "--parent-pid", str(os.getpid())]

    if return_string:
        return model_path_args + " " + " ".join(extra_args)

    return model_path_args + extra_args


def create_context():
    context = local()

    # temp hack, throws an attribute not found error otherwise
    context.torch_device = get_device(0)
    context.device = f"{context.torch_device.type}:{context.torch_device.index}"
    context.half_precision = True
    context.vram_usage_level = None

    context.models = {}
    context.model_paths = {}
    context.model_configs = {}
    context.device_name = None
    context.vram_optimizations = set()
    context.vram_usage_level = "balanced"
    context.test_diffusers = False
    context.enable_codeformer = False

    return context


def do_start_backend(was_still_installing, run_fn):
    global WEBUI_HOST, WEBUI_PORT

    config = getConfig()
    backend_config = config.get("backend_config") or {}

    WEBUI_HOST = backend_config.get("host", "localhost")
    WEBUI_PORT = backend_config.get("port", "7860")

    def restart_if_webui_dies_after_starting():
        has_started = False

        while True:
            if backend_process is None:
                return

            # Check if the process is actually dead
            return_code = backend_process.poll()

            if return_code is not None:
                # Process has terminated
                if has_started:
                    print(f"######################## Backend process died with code {return_code}. Restarting...")
                    stop_backend()
                    backend_thread = Thread(target=target)
                    backend_thread.start()
                    break
                else:
                    # Process died before starting successfully
                    print(f"######################## Backend process failed to start (exit code {return_code})")
                    break
            elif not has_started:
                # Process is running, check if it has started successfully via ping
                try:
                    ping(timeout=5)
                    has_started = True

                    if was_still_installing:
                        ui = config.get("ui", {})
                        net = config.get("net", {})
                        port = net.get("listen_port", 9000)

                        if ui.get("open_browser_on_start", True):
                            import webbrowser

                            log.info("Opening browser..")

                            webbrowser.open(f"http://localhost:{port}")
                except (TimeoutError, ConnectionError):
                    pass  # Still starting up
                except Exception:
                    import traceback

                    log.exception(traceback.format_exc())

            time.sleep(1)

    def target():
        global backend_process

        backend_process = run_fn()

        # atexit.register isn't 100% reliable, that's why we also use `forge_monitor_parent_process.patch`
        # which causes Forge to kill itself if the parent pid passed to it is no longer valid.
        atexit.register(backend_process.terminate)

        restart_if_dead_thread = Thread(target=restart_if_webui_dies_after_starting)
        restart_if_dead_thread.start()

        backend_process.wait()

    backend_thread = Thread(target=target)
    backend_thread.start()


def stop_backend():
    global backend_process

    if backend_process:
        try:
            kill(backend_process.pid)
        except:
            pass

    backend_process = None
