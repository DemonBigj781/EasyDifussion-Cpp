# Native model offload versus streaming

CPU offload decides where parameter weights are backed. It does not on its
own limit how many of those weights a GPU forward stages. The old app also
accepted layer streaming while the planner had no VRAM budget, so that path
could still stage a whole forward's parameters. For the native application:

- `--offload-to-cpu --stream-layers` requests host-backed denoiser streaming.
- `--video-offload-to-cpu --video-stream-layers` applies only to native video.
  General CPU offload also satisfies the video parameter requirement.
- Streaming requires one accelerator, CPU-resident denoiser parameters, and a
  graph with usable cut points. It does not change multi-device layer splitting.
- The native runner uses reported free device memory minus its existing 512 MiB
  reserve as **planner headroom**, not a hard VRAM limit. Workspace estimates,
  driver accounting and competing allocations still matter. It rejects unusable
  telemetry and CUDA unified-memory host reporting instead of treating zero
  as unlimited.
- If the denoiser cannot produce a valid bounded plan, it fails instead of
  staging the entire graph silently. The allocator still may reject a graph a
  planner estimates will fit. ControlNet multi-output, conditioning, and
  decoding are not part of this denoiser-streaming guarantee.
- Native app flags `--max-vram` and `--video-max-vram` were retired. Remove them
  from saved startup arguments for new app binaries. Native library budget
  fields and other executables' arguments are not changed.
- In streaming contexts, eager loading and retained accelerator staging are
  disabled even with `--keep-model-loaded`. Host parameter storage can still
  be reused; the executor releases accelerator staging between streamed
  denoiser segments. Nonstreaming contexts retain their previous behavior.

A weight-free policy test checks activation, shrinking telemetry and exact
budget arithmetic (`ctest -R '^native-streaming-policy$'`). The separate
`model-weight-residency-cpu` fixture exercises actual host-parameter load,
stage, readback and release across two CPU backend instances. Neither proves
GPU memory savings or that an oversized model generates correctly on hardware.
