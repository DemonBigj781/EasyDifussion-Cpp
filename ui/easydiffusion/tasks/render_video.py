import json
import time

from easydiffusion import model_manager, runtime
from easydiffusion.types import ModelsData, OutputFormatData, TaskData, VideoGenerationRequest
from easydiffusion.utils import log
from easydiffusion.utils.model_identifier import identify_model_type, read_gguf_header, read_safetensors_header
from easydiffusion.video_companions import discover_mochi_companions
from easydiffusion.video_output import build_animatediff_gif

from .task import Task


class VideoTask(Task):
    """Queued native video frame generation for the sdkit3 backend."""

    def __init__(
        self,
        req: VideoGenerationRequest,
        task_data: TaskData,
        models_data: ModelsData,
        output_format: OutputFormatData,
    ):
        super().__init__(task_data.session_id)
        task_data.request_id = self.id
        self.request = req
        self.task_data = task_data
        self.models_data = models_data
        self.output_format = output_format

    def run(self):
        from easydiffusion import task_manager
        from easydiffusion.backend_manager import backend

        if not hasattr(backend, "generate_video"):
            raise RuntimeError("The selected backend does not provide native video generation")

        context = runtime.context
        if self.request.video_steps is not None:
            self.request.num_inference_steps = self.request.video_steps

        def cancellation_requested(forward_to_backend=False):
            cancelled = isinstance(task_manager.current_state_error, (SystemExit, StopAsyncIteration)) or isinstance(
                self.error, StopAsyncIteration
            )
            if not cancelled:
                return False
            if forward_to_backend:
                backend.stop_rendering(context)
            if isinstance(task_manager.current_state_error, StopAsyncIteration):
                self.error = task_manager.current_state_error
                task_manager.current_state_error = None
                log.info(f"Session {self.session_id} sent cancel signal for video task {self.id}")
            return True

        task_manager.current_state = task_manager.ServerStates.LoadingModel
        log.info(f"Video task {self.id}: requested model components: {self.models_data.model_paths}; motion-module={self.request.motion_module_model!r}")
        try:
            model_manager.resolve_model_paths(self.models_data)
        except Exception:
            log.exception(f"Video task {self.id}: model path resolution failed: {self.models_data.model_paths}")
            raise

        checkpoint_path = self.models_data.model_paths.get("stable-diffusion")
        if self.request.motion_module_model:
            self.request.motion_module_model = model_manager.resolve_model_to_use(
                self.request.motion_module_model, model_type="motion-module")

        model_class = identify_model_type(checkpoint_path) if checkpoint_path else None
        if self.request.motion_module_model and checkpoint_path and model_class and model_class.startswith(("sd_v1", "sd_xl")):
            path = str(checkpoint_path)
            header = read_gguf_header(path) if path.lower().endswith(".gguf") else read_safetensors_header(path)
            keys = tuple(header)
            embedded_vae = any(k.startswith(("first_stage_model.", "vae.")) for k in keys)
            embedded_clip = any(k.startswith(("cond_stage_model.", "conditioner.embedders.", "text_encoders.", "clip_l.", "clip_g.")) for k in keys)
            missing = []
            if not embedded_vae and not self.models_data.model_paths.get("vae"):
                missing.append("VAE (select a compatible VAE under Options)")
            if not embedded_clip and not self.models_data.model_paths.get("text-encoder"):
                missing.append("CLIP text encoder (select CLIP-L for SD 1.5 or compatible SDXL encoders under Options)")
            if missing:
                message = f"AnimateDiff checkpoint {path} is missing required model components: " + "; ".join(missing)
                log.error(message)
                raise RuntimeError(message)

        if model_class in ("ltx2", "ltx2_3"):
            required = (
                ("video VAE", self.models_data.model_paths.get("vae")),
                ("Gemma 3 text encoder", self.models_data.model_paths.get("text-encoder")),
                ("audio VAE", self.request.audio_vae_model),
                ("embedding connectors", self.request.embeddings_connectors_model),
            )
            missing = [label for label, value in required if not value]
            if missing:
                raise RuntimeError(f"{model_class} is missing {', '.join(missing)}; select its resources before generating")
            self.request.audio_vae_model = model_manager.resolve_model_to_use(
                self.request.audio_vae_model, model_type="vae"
            )
            self.request.embeddings_connectors_model = model_manager.resolve_model_to_use(
                self.request.embeddings_connectors_model, model_type="text-encoder"
            )
            self.request.sampler_name = "euler"
            self.request.scheduler_name = "ltx2"
            if "distilled" in str(checkpoint_path).lower():
                self.request.num_inference_steps = 8
                self.request.guidance_scale = 1.0
        if model_class and model_class.startswith("ltx_video"):
            # Original LTX-Video uses the Euler linear/quadratic flow schedule.
            # Enforce it at the API boundary as well as in the browser UI.
            self.request.sampler_name = "euler"
            self.request.scheduler_name = "linear_quadratic"
            if "distilled" in str(checkpoint_path).lower():
                self.request.num_inference_steps = 8
                self.request.guidance_scale = 1.0
        if model_class == "mochi_v1_preview":
            discovered = discover_mochi_companions(checkpoint_path)
            for model_type, model_path in discovered.items():
                if not self.models_data.model_paths.get(model_type):
                    self.models_data.model_paths[model_type] = model_path
                    log.info(f"Auto-selected Mochi {model_type}: {model_path}")
            missing = [
                label
                for label, model_type in (("Video VAE", "vae"), ("Video Text Encoder", "text-encoder"))
                if not self.models_data.model_paths.get(model_type)
            ]
            if missing:
                raise RuntimeError(
                    f"Mochi could not auto-detect {' and '.join(missing)} beside {checkpoint_path}"
                )
            if self.request.init_image or self.request.end_image:
                raise RuntimeError("Native Mochi currently supports text-to-video only")
            # The Genmo linear/quadratic sigma schedule is part of Mochi's
            # inference recipe. Enforce it for API clients as well as the UI.
            self.request.sampler_name = "euler"
            self.request.scheduler_name = "mochi"

        log.info(f"Video task {self.id}: resolved components: {self.models_data.model_paths}; motion-module={self.request.motion_module_model!r}")
        try:
            model_manager.reload_models_if_necessary(context, self.models_data)
            model_manager.fail_if_models_did_not_load(context)
        except Exception:
            log.exception(f"Video task {self.id}: model setup failed: {self.models_data.model_paths}")
            raise
        if cancellation_requested():
            return

        task_manager.current_state = task_manager.ServerStates.Rendering
        last_callback = [None]

        def progress(
            _images,
            step,
            total_steps=None,
            completed=False,
            phase="sampling",
            phase_current=0,
            phase_total=0,
        ):
            if cancellation_requested(forward_to_backend=True):
                return
            now = time.monotonic()
            step_time = -1 if last_callback[0] is None else now - last_callback[0]
            last_callback[0] = now
            total = max(1, int(total_steps or self.request.num_inference_steps))
            effective_phase = "finalizing" if completed else phase or ("decoding" if step >= total else "sampling")
            self.buffer_queue.put(
                json.dumps(
                    {
                        "step": int(step),
                        "total_steps": total,
                        "phase": effective_phase,
                        "phase_current": max(0, int(phase_current or 0)),
                        "phase_total": max(0, int(phase_total or 0)),
                        "step_time": step_time,
                    }
                )
            )
            task_manager.keep_task_alive(self)

        try:
            request_args = self.request.dict()
            request_args.pop("video_steps", None)
            result = backend.generate_video(context, callback=progress, **request_args)
        except Exception:
            if cancellation_requested():
                return
            raise
        if cancellation_requested():
            return
        frames = result.get("frames") or []
        native_gif = result.get("gif") if self.request.motion_module_model else None
        frame_count = int(result.get("frame_count") or len(frames))
        if not frames and not (native_gif and frame_count > 0):
            raise RuntimeError("The native video backend returned no frames")

        request_data = self.request.dict()
        request_data["init_image"] = None
        request_data["end_image"] = None
        fps = int(result.get("fps") or self.request.fps)
        response_format = self.output_format.dict()
        video_info = {"fps": fps, "frame_count": frame_count, "cache_mode": self.request.cache_mode}
        if self.request.motion_module_model:
            if native_gif:
                gif, gif_info = native_gif, {"frame_count": frame_count}
            else:
                gif, gif_info = build_animatediff_gif(frames, fps, log)
            request_data["output_format"] = "gif"
            response_format["output_format"] = "gif"
            output = [{"data": gif, "seed": self.request.seed, "fps": fps,
                       "frame_count": frame_count, "encoded_frame_count": gif_info["frame_count"],
                       "is_video": True, "is_animated_gif": True}]
            video_info.update(format="gif", is_animated_gif=True)
        else:
            output = [
                {
                    "data": frame,
                    "seed": self.request.seed,
                    "frame_index": index,
                    "fps": int(result.get("fps") or self.request.fps),
                    "is_video_frame": True,
                }
                for index, frame in enumerate(frames)
            ]
        self.response = {
            "status": "succeeded",
            "render_request": request_data,
            "task_data": {
                **self.task_data.dict(),
                **response_format,
            },
            "video": video_info,
            "output": output,
        }
        self.buffer_queue.put(json.dumps(self.response))
        log.info(f"Native video task completed with {frame_count} frames")
