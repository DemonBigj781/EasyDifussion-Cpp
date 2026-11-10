# 02 — Additional Video Model Compatibility

## Objective
Generalize the native video pipeline so new model families can be added without copying an entire backend implementation.

## Implementation
1. Audit current assumptions about tensor names, latent shape, temporal axis, frame count, scheduler, text/image conditioning, VAE layout, FPS metadata, and output packing.
2. Introduce a `VideoModelDescriptor` containing architecture/family, latent channels, temporal behavior, required encoders, VAE type, scheduler compatibility, and supported dtypes.
3. Detect model family from metadata/config and tensor signatures rather than filename alone.
4. Put architecture-specific tensor-name translation and special transforms behind per-family adapters.
5. Keep generic pipeline phases shared: load → condition → initialize latents → sample → decode → encode output.
6. Make frame count, width/height constraints, temporal compression, and FPS capability-driven.
7. Add one additional model family first to prove the abstraction before adding more.

## Deferred follow-ups
These are later work, separate from the current SD1.5/SDXL video toggle, mutually exclusive EasyCache/TeaCache selection, and validated GIF output/display.

- [ ] Evaluate AnimateDiff SD2 support using https://github.com/damian0815/AnimateDiff-sd2 as a reference. Verify matching SD2 motion weights, tensor mappings, conditioning, and prediction/scheduler behavior in the native C++ backend before expanding UI model-family eligibility. Do not assume SD1.5 motion adapters are interchangeable with SD2.
- [ ] Evaluate https://github.com/N3oRay/AnimateHYBRID for reusable animation techniques. Audit actual motion-weight execution, dependencies, licensing, and reproducible quality/memory results before deciding whether to port anything; do not treat README claims or synthetic noise as verified motion-module inference.

## Likely code areas
`source/sdkit3-port-source/`, stable-diffusion.cpp video model loading, server request schema, UI video controls, model manager, and tests.

## Dependencies
Backend/device capability reporting should be available so unsupported model/backend combinations can be rejected before allocation.

## Fallback
An unknown video architecture should return an explicit unsupported-model error and must not be misrouted through an image model path.

## Validation
Golden test model metadata parsing, tensor mapping, short low-resolution generation, VAE decode, frame count, FPS metadata, cancellation, and low-memory mode.

## Complete when
At least two materially different video architecture families execute through the same top-level pipeline interface without architecture-specific logic leaking throughout the server/UI.
