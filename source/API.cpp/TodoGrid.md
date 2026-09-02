# API.cpp implementation grid

Each feature is followed by its normalized Common method files. Backend cells track implementation and validation maturity rather than only whether a compiler accepted the code.

## Status legend

A cell may advance through the following states without implying that later stages have been proven:

| Mark | State | Meaning |
| --- | --- | --- |
| blank | Not established | No implementation or validation state has been established yet. |
| `I` | Implemented | An implementation exists, but it may still contain missing paths, placeholders, or incomplete behavior. |
| `C` | Code complete | The intended feature contract is implemented and no known required code paths remain unfinished. |
| `B` | Builds | The applicable backend/toolchain has successfully compiled the implementation. This does not imply that it has executed correctly on real hardware. |
| `R` | Runtime tested | The implementation has successfully executed relevant runtime tests on applicable hardware/runtime. |
| `E` | End-to-end tested | The complete definition -> translation -> Common -> Library -> Easy Diffusion route has successfully executed for the intended workflow. |
| `XXX` / `XXXX` | Proven complete | The implementation has been thoroughly accounted for end to end, including the intended feature paths, integration, relevant edge/error behavior, cleanup/ownership/fallback behavior where applicable, and required validation. Existing marked cells retain this meaning. |

Status is progressive: `I -> C -> B -> R -> E -> proven complete`. A later state includes the expectations of the earlier states, but a successful build or narrow runtime test must never be interpreted as proof that the implementation itself is complete. Likewise, an end-to-end happy-path result is not sufficient for `XXX` / `XXXX` unless the feature's intended behavior has been thoroughly accounted for.

All cells that were already marked `XXX` / `XXXX` before this legend was introduced are intentionally preserved as proven-complete results; this change does not downgrade or reinterpret them.

## Detect

`detect.cpp` discovers compute backends and runtime-visible devices. It reports
availability, identity, device class, architecture or compute version when
available, and a point-in-time memory snapshot. Model-format, ControlNet, and
image-object detection are separate concerns.

```text
===========================================================================================
[              ][cpu][cuda][rocm][oneapi][opencl][openvino][opengl][vulkan][mesa][directml]
===========================================================================================
[detect        ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[detect.cpp    ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
```

Source routes and compiler-only workflows now exist for every column. The
workflow assignments are CPU `001`, CUDA `011`, ROCm `021`, oneAPI `031`,
OpenCL `041`, OpenVINO `051`, Vulkan `101`, OpenGL `111`, DirectML `121`, and
Mesa `131`. Intermediate states may now be recorded as work progresses; only
`XXX` / `XXXX` denotes that the backend/feature has been thoroughly proven.

## Load

load.cpp validates a component payload, creates backend-owned storage, copies
the payload into that storage, and returns a normalized owned resource. A load
cell does not imply component parsing or graph construction unless that
component's contract explicitly adds those operations.

~~~text
===========================================================================================
[              ][cpu][cuda][rocm][oneapi][opencl][openvino][opengl][vulkan][mesa][directml]
===========================================================================================
[clip          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[clip-vision   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[condition     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[image         ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[latent        ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[mask          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[model         ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unet          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[vae           ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
~~~

## Unload

unload.cpp releases a resource through the backend that owns it and clears
the normalized resource only after release succeeds. Backend mismatch,
double-unload, and empty-handle behavior are validation requirements.

~~~text
===========================================================================================
[              ][cpu][cuda][rocm][oneapi][opencl][openvino][opengl][vulkan][mesa][directml]
===========================================================================================
[clip          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[clip-vision   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[condition     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[image         ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[latent        ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[mask          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[model         ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unet          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[vae           ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
~~~

## Overflow

`plan.cpp` builds the backend-qualified fallback order. `allocate.cpp` executes
only candidates owned by its backend, and `release.cpp` frees through the exact
allocator recorded in the normalized resource. CPU heap overflow and CUDA
active/secondary VRAM, Managed Memory, and mapped-host RAM are implemented.

~~~text
===========================================================================================
[              ][cpu][cuda][rocm][oneapi][opencl][openvino][opengl][vulkan][mesa][directml]
===========================================================================================
[overflow      ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[plan.cpp      ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
[allocate.cpp  ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
[release.cpp   ][XXX][XXXX][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
~~~

CPU allocation, ownership rejection, release, and a synthetic mixed-backend
plan passed locally. CUDA compiler and runtime validation passed on an RTX 3060
(`sm_86`, driver 580.94.18, CUDA toolkit 12.4): forced overflow selected
`cudaMallocManaged`; the bounded stress test then retained 1536 MiB free VRAM,
provoked a real 2048 MiB `cudaMalloc` OOM, selected CPU-preferred Managed
Memory, executed a CUDA kernel through it, verified the pages on the CPU, and
released all allocations. The mapped-host path compiled and remains the runtime
fallback when concurrent Managed Memory is absent.

The CUDA secondary route currently covers only runtime-visible CUDA GPUs.
ROCm/OpenCL/Vulkan AMD execution and oneAPI/OpenCL Intel execution remain blank.
The planned H3C XG310 test must enumerate four independent low-power Intel SG1
devices with 8 GB each, not one contiguous 32 GB device. `gpu-zram`, zram, and
swap also remain unimplemented.

## Attention

```text
===========================================================================================
[              ][cpu][cuda][rocm][oneapi][opencl][openvino][opengl][vulkan][mesa][directml]
===========================================================================================
[xformer       ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[av.cpp        ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[forward.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[mask.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[qkt.cpp       ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[softmax.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[sage          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[support.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[key_mean.cpp  ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[quantize.cpp  ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[qkt.cpp       ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[mask.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[softmax.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[sv.cpp        ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[normalize.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[forward.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[flash         ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[support.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[workspace.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[select.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[qkt.cpp       ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[mask.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[softmax.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[av.cpp        ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[combine.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[forward.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[flex          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[mask_size.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[validate.cpp  ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[sum_heads.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[margin.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[normalize.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[threshold.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[pool.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[expand.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[select.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[split         ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[plan.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[split.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[qkt.cpp       ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[mask.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[softmax.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[av.cpp        ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[merge.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[forward.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================


===========================================================================================
[              ][cpu][cuda][rocm][oneapi][opencl][openvino][opengl][vulkan][mesa][directml]
===========================================================================================
[easy          ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[support.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[validate.cpp  ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[reset.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[init.cpp      ][   ][    ][    ][      ][      ][