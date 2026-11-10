# API runtime test applications

This tree contains standalone executables for normalized DiffUser.cpp routes. A feature test calls its Common API and links the selected backend translation and definition; test code must not call backend implementation symbols directly.

## xFormers test layout

The xFormers suite has two groups under `MultiTest/xformers/`:

- `cycle/` validates registration, capabilities, analytical results, rejected inputs, GQA/MQA head mapping, masks, ALiBi, soft-cap behavior, and repeated custom Common model load/forward/unload cycles.
- `generate-image/` retains the historical stable-diffusion.cpp integration
  harness. The current isolated engine snapshot no longer contains the
  GGML-to-DiffUser adapters or CUDA observation hooks, so its CUDA generators
  are not current build or end-to-end targets. The CPU generator is an engine
  smoke program and does not exercise DiffUser Common.

Build the current Common cycle programs without running them:

```sh
make xformers-cpu
make xformers-cuda
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

The canonical cycle artifacts are `build/xformers-load-unload-cpu-test` and
`build/xformers-load-unload-cuda-test`. The retained image harness and its
historical options are documented in
`MultiTest/xformers/generate-image/README.md`.

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
the unwired Flash/Sage/Split `MultiTest` families. CPU Flex has a public Common
feature test under `Feature/Attention/Flex/Cpu`; build or run it with
`make flex-cpu` or `make run-flex-cpu`.

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
