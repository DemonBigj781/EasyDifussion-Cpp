import json
import time

from easydiffusion import model_manager, runtime
from easydiffusion.types import ModelsData, OutputFormatData, TaskData, VideoGenerationRequest
from easydiffusion.utils import log
from easydiffusion.utils.model_identifier import identify_model_type
from easydiffusion.video_companions import discover_mochi_companions

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
        model_manager.resolve_model_paths(self.models_data)

        checkpoint_path = self.models_data.model_paths.get("stable-diffusion")
        model_class = identify_model_type(checkpoint_path) if checkpoint_path else None
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

        model_manager.reload_models_if_necessary(context, self.models_data)
        model_manager.fail_if_models_did_not_load(context)
        if cancellation_requested():
            return

        task_manager.current_state = task_manager.ServerStates.Rendering
        last_callback = [None]

        def progress(_images, step, *_args):
            if cancellation_requested(forward_to_backend=True):
                return
            now = time.monotonic()
            step_time = -1 if last_callback[0] is None else now - last_callback[0]
            last_callback[0] = now
            self.buffer_queue.put(
                json.dumps(
                    {
                        "step": int(step),
                        "total_steps": int(self.request.num_inference_steps),
                        "step_time": step_time,
                    }
                )
            )
            task_manager.keep_task_alive(self)

        try:
            result = backend.generate_video(context, callback=progress, **self.request.dict())
        except Exception:
            if cancellation_requested():
                return
            raise
        if cancellation_requested():
            return
        frames = result.get("frames") or []
        if not frames:
            raise RuntimeError("The native video backend returned no frames")

        request_data = self.request.dict()
        request_data["init_image"] = None
        request_data["end_image"] = None
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
                **self.output_format.dict(),
            },
            "video": {
                "fps": int(result.get("fps") or self.request.fps),
                "frame_count": len(output),
                "cache_mode": self.request.cache_mode,
            },
            "output": output,
        }
        self.buffer_queue.put(json.dumps(self.response))
        log.info(f"Native video task completed with {len(output)} frames")
