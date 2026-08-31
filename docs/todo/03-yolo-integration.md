# 03 — YOLO Integration

## Objective
Add object detection as a reusable Easy Diffusion service for image analysis, automatic masks, tagging, routing, and video-frame inspection.

## Implementation
1. Define a detector-neutral interface: model load, preprocess, infer, postprocess, unload.
2. Represent output as structured detections: class ID/name, confidence, normalized and pixel bounding box, and optional segmentation mask.
3. Support a bounded first runtime such as ONNX Runtime or OpenVINO; later add native/GGML execution only if beneficial.
4. Implement letterboxing/resizing, normalization, channel/layout conversion, model invocation, output decoding, confidence filtering, and NMS.
5. Expose detector settings through API/UI: model, device, confidence threshold, IoU threshold, selected classes, and output action.
6. Add pipeline consumers: create masks from detections, add tags/metadata, select regions for inpainting, or feed detections to caption/prompt logic.
7. For video, allow frame sampling and batching so detection does not automatically run on every frame.

## Dependencies
OpenVINO support can provide a good Intel path but must not be mandatory. A CPU path should exist.

## Security / data handling
Run local models locally by default. Do not upload user images to external services for detection unless a separate provider explicitly requires and discloses it.

## Validation
Use known test images with expected class/box tolerances, verify NMS, resized coordinates, masks if present, CPU and accelerator paths, and pipeline use.

## Complete when
A user can invoke YOLO on an image, receive stable structured detections, and use those detections in at least one generation/edit pipeline action.
