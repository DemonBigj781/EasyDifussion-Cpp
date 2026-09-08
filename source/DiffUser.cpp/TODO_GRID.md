# DiffUser.cpp implementation grid

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
| `E` | End-to-end tested | The complete Easy Diffusion/application -> Common -> translation -> definition -> native runtime/device route has successfully executed for the intended workflow. `library/` may package the Common ABI but is not a routing layer. |
| `N/A` | Deliberately not separate | The method is intentionally fused into another operation or otherwise does not exist as a separate backend callback. |
| `XXX` / `XXXX` | Proven complete | The implementation has been thoroughly accounted for end to end, including the intended feature paths, integration, relevant edge/error behavior, cleanup/ownership/fallback behavior where applicable, and required validation. Existing marked cells retain this meaning. |

Status is progressive: `I -> C -> B -> R -> E -> proven complete`. A later state includes the expectations of the earlier states, but a successful build or narrow runtime test must never be interpreted as proof that the implementation itself is complete. Likewise, an end-to-end happy-path result is not sufficient for `XXX` / `XXXX` unless the feature's intended behavior has been thoroughly accounted for.

All cells that were already marked `XXX` / `XXXX` before this legend was introduced are intentionally preserved as proven-complete results; this change does not downgrade or reinterpret them. When a route changes after that proof, the [dated Common-routing audit](../../Audit/2026-09-06/COMMON_ROUTING_AUDIT.md) records any architectural revalidation still required; a historical proof mark does not waive a newer routing defect.

## Detect

`detect.cpp` discovers compute backends and runtime-visible devices. It reports
availability, identity, device class, architecture or compute version when
available, and a point-in-time memory snapshot. Model-format, ControlNet, and
image-object detection are separate concerns.

```text
===========================================================================================
[              ][cpu][cuda][rocm][oneapi][opencl][openvino][opengl][vulkan][mesa][directml]
===========================================================================================
[detect        ][R  ][R   ][I   ][I     ][R     ][I       ][R     ][R     ][R   ][I       ]
===========================================================================================
[detect.cpp    ][R  ][R   ][I   ][I     ][R     ][I       ][R     ][R     ][R   ][I       ]
===========================================================================================
```

Source routes and compiler-only workflows now exist for every column. CPU,
CUDA, OpenCL, Vulkan, OpenGL, and Mesa were rebuilt and runtime-tested on
2026-09-05. The remaining backends are marked implemented from source audit,
but their unavailable local toolchains/runtimes prevent stronger claims. The
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
[model         ][R  ][R   ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[load.cpp      ][R  ][R   ][    ][      ][      ][        ][      ][      ][    ][        ]
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
[model         ][R  ][R   ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[unload.cpp    ][R  ][R   ][    ][      ][      ][        ][      ][      ][    ][        ]
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
[overflow      ][R  ][R   ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[plan.cpp      ][R  ][R   ][    ][      ][      ][        ][      ][      ][    ][        ]
[allocate.cpp  ][R  ][R   ][    ][      ][      ][        ][      ][      ][    ][        ]
[release.cpp   ][R  ][R   ][    ][      ][      ][        ][      ][      ][    ][        ]
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
[xformers      ][R  ][E   ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[av.cpp        ][R  ][N/A ][    ][      ][      ][        ][      ][      ][    ][        ]
[forward.cpp   ][R  ][E   ][    ][      ][      ][        ][      ][      ][    ][        ]
[mask.cpp      ][R  ][N/A ][    ][      ][      ][        ][      ][      ][    ][        ]
[qkt.cpp       ][R  ][N/A ][    ][      ][      ][        ][      ][      ][    ][        ]
[softmax.cpp   ][R  ][N/A ][    ][      ][      ][        ][      ][      ][    ][        ]
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
[flash         ][B  ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[support.cpp   ][B  ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[workspace.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[select.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[qkt.cpp       ][N/A][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[mask.cpp      ][N/A][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[softmax.cpp   ][N/A][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[av.cpp        ][N/A][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[combine.cpp   ][N/A][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[forward.cpp   ][B  ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
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
```

CUDA xFormers deliberately exposes one fused `forward` callback to Common, so
its materialized QKT, mask, softmax, and AV callbacks are `N/A`. The same
registered Common translation now serves both standalone Common tests and the
GGML application adapter. CPU FlashAttention has an upstream Common route that
builds and passes a native-boundary smoke test; it is not yet numerical runtime
proof.

## Cache

EasyCache and TeaCache are planned method inventories. Blank cells do not claim
source, build wiring, or support.

```text
===========================================================================================
[              ][cpu][cuda][rocm][oneapi][opencl][openvino][opengl][vulkan][mesa][directml]
===========================================================================================
[easycache     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[support.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[validate.cpp  ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[reset.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[init.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[enabled.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[sigma.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[begin_step.cpp][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[active.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[skipped.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[has_cache.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[store.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[apply.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[before.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[after.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[teacache      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
[support.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[configure.cpp ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[validate.cpp  ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[init.cpp      ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[enabled.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[begin_step.cpp][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[rel_l1.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[rescale.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[accumulate.cpp][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[reuse.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[store.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[apply.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[skipped.cpp   ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[before.cpp    ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
[after.cpp     ][   ][    ][    ][      ][      ][        ][      ][      ][    ][        ]
===========================================================================================
```
