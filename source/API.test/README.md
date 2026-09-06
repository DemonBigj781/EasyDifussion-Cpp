# API runtime test applications

This tree contains standalone executables for normalized API.cpp routes. A feature test calls its Common API and links the selected backend translation and definition; test code must not call backend implementation symbols directly.

## xFormers test layout

The xFormers suite has two complementary groups under `MultiTest/xformers/`:

- `cycle/` validates registration, capabilities, analytical results, rejected inputs, GQA/MQA head mapping, masks, ALiBi, soft-cap behavior, and repeated custom Common model load/forward/unload cycles.
- `generate-image/` builds real CPU/CUDA image-generation programs around stable-diffusion.cpp. They generate one 512x512 image, request memory-efficient attention, and lock sampling to DDIM trailing with the Simple scheduler. The CUDA program additionally fails unless the native xFormers kernel actually launches.

Build the four programs without running them:

```sh
make xformers-cpu xformers-cpu-gen
make xformers-cuda xformers-cuda-gen
```

Run the cycle programs:

```sh
make run-xformers-cpu
make run-xformers-cuda
```

Or build and run both cycle programs:

```sh
make check-xformers
```

The canonical cycle artifacts are `build/xformers-load-unload-cpu-test` and `build/xformers-load-unload-cuda-test`. The image-generation artifacts are `build/xformers-cpu-gen` and `build/xformers-cuda-gen`; they require `-m` and `-p` arguments and are documented in `MultiTest/xformers/generate-image/README.md`.

The CUDA Common route requires an NVIDIA Pascal-or-newer device, CUDA
development headers, `nvcc`, the CUDA runtime, and device access. It accepts
strided device or managed-memory F32/F16/BF16 Q/K/V with F32 output, supported
mask dtypes/broadcasting, ALiBi, soft-cap, attention sinks, and an opaque CUDA
stream. A restricted container may compile successfully while failing at
runtime because no GPU is exposed.

## Other API tests

Detect programs use `make cpu`, `cuda`, `opencl`, `vulkan`, `opengl`, or
`mesa`. Model-byte lifecycle programs use `make load-cpu`, `load-cuda`,
`unload-cpu`, or `unload-cuda`. Overflow programs use `make overflow-cpu` or
`overflow-cuda`. Add the corresponding `run-` prefix to execute a target.

Backend-specific test directories or blank files are reservations until an
actual target selects them. This currently applies to Mesa/OpenGL/Vulkan
load/unload and xFormers placeholders, Mesa/OpenGL Overflow directories, and
the unwired Flash/Flex/Sage/Split `MultiTest` families.

Load/unload programs read `MODELS/lifecycle-model.fixture` by default. Pass another model or binary payload as the first argument to exercise a larger local fixture.

## GitHub and Vast.ai

Every numbered API workflow offers `Compile Test Code` without requesting paid
hardware. `Test, Vast AI` is reserved for a backend/runtime that cannot be
tested locally: it stages the GitHub-built executable, requires exact
`APPROVE` plus a numeric offer ID, and runs that same executable remotely
without compiling. The shared reusable workflow is
`.github/workflows/@@@-vast-ai-runtime-test.yaml`. `VASTAI_KEY` is consumed
only from the repository secret and must never enter source, artifacts, images,
commands, or logs. Vast.ai is a runtime target, not a build host.
