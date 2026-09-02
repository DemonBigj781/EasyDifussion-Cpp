# Detect

`features/detect/` owns compute-backend and device discovery. It does not own
model-format inspection, ControlNet preprocessing, or image-object detection.

The normalized operation is `detect.cpp`. A result reports backend
availability and runtime-visible devices, including device class, stable index,
name, architecture, compute version when the backend exposes one, and a
point-in-time total/free-memory snapshot. Detection must not retain a native
runtime object, allocate model memory, or silently select a device for later
work.

The route is:

```text
[backend]/definition/[device-type]/detect.cpp
    -> [backend]/translation/[device-type]/detect.cpp
    -> common/detect.cpp
    -> Library backend handler
    -> Easy Diffusion
```

Source routes now exist for CPU, CUDA, ROCm, oneAPI, OpenCL, OpenVINO, OpenGL,
Vulkan, Mesa, and DirectML. Each route has a Library handler and an assigned
compiler-only workflow. The workflow proves source compatibility only; it does
not prove that hardware was present or that runtime enumeration succeeded.

Backend constraints are preserved instead of replaced with guessed values:

- oneAPI, OpenCL, and OpenVINO split discovery into CPU, GPU, and NPU or
  accelerator device classes;
- Vulkan and Mesa split CPU/software and hardware GPU renderers;
- OpenGL and Mesa require a current OpenGL context and never create a hidden
  context as a side effect of detection;
- APIs without a portable live-free-memory query leave `free_memory` at zero;
- DirectML reports dedicated adapter memory only, avoiding double-counting
  system RAM on integrated hardware.

TodoGrid support cells remain blank until the applicable compiler workflow has
passed and separate runtime-device validation is recorded.
