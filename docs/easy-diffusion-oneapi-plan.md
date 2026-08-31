# Easy Diffusion oneAPI / SYCL Expansion Plan

## Status

Design notes / implementation plan only. This document does not represent completed functionality.

## Scope

This note covers expanding oneAPI / SYCL support through the **Easy Diffusion image-generation path** in EasyDifussion-Cpp.

The focus is specifically on:

- `sdkit3-port-source`
- `stable-diffusion.cpp`
- GGML execution used by Easy Diffusion
- image-generation model loading
- UNet / DiT execution
- CLIP and text encoders
- VAE encode/decode
- ControlNet
- LoRA application
- attention kernels
- tensor offload and streaming
- VRAM budgeting
- multi-GPU scheduling
- device discovery and selection

This does **not** make llama.cpp / TIPO part of the same runtime design. llama.cpp already has its own backend system and can continue using that separately.

## Existing advantage

The repository already vendors `ggml-sycl` code inside both llama.cpp and stable-diffusion.cpp. For Easy Diffusion, the important part is the stable-diffusion.cpp side.

Therefore the preferred implementation is **not** to create a new oneAPI tensor engine from scratch.

Instead:

```text
Easy Diffusion
    -> sdkit3
        -> stable-diffusion.cpp
            -> GGML
                -> CPU
                -> CUDA
                -> SYCL / oneAPI
```

The work should center on enabling, exposing, testing, and extending the existing SYCL backend throughout Easy Diffusion.

## Hardware target

The intended accelerator is the H3C XG310 board, which exposes multiple Intel GPU devices rather than one unified VRAM pool.

The implementation must treat each GPU as an independent SYCL device with its own memory budget.

Conceptually:

```text
XG310
  ├── SYCL GPU 0
  ├── SYCL GPU 1
  ├── SYCL GPU 2
  └── SYCL GPU 3
```

Do not assume that aggregate board memory can be allocated as a single contiguous device buffer.

## Backend selection

Easy Diffusion should expose an explicit backend concept rather than relying on build-time behavior alone.

Suggested runtime enum:

```cpp
enum class ComputeBackend {
    Auto,
    CPU,
    CUDA,
    SYCL,
};
```

Suggested configuration:

```cpp
struct BackendConfig {
    ComputeBackend backend = ComputeBackend::Auto;
    std::vector<int> device_ids;
    bool allow_cpu_fallback = true;
};
```

Runtime policy:

```text
Auto
 ├── CUDA available -> CUDA
 ├── SYCL GPU available -> SYCL
 └── CPU
```

Explicit `SYCL` should fail clearly if no usable SYCL GPU exists unless CPU fallback is enabled.

## Device discovery

Add a native device enumeration layer in sdkit3.

Suggested API:

```cpp
struct DeviceInfo {
    int logical_id = -1;
    std::string backend;
    std::string name;
    uint64_t global_memory = 0;
    uint64_t max_allocation = 0;
    bool is_gpu = false;
};

std::vector<DeviceInfo> enumerate_compute_devices();
```

For SYCL builds, enumerate available devices and retain stable logical IDs for the running process.

The UI should eventually be able to display something similar to:

```text
SYCL 0 - Intel Server GPU - 8 GB
SYCL 1 - Intel Server GPU - 8 GB
SYCL 2 - Intel Server GPU - 8 GB
SYCL 3 - Intel Server GPU - 8 GB
```

## Build-system integration

The Easy Diffusion native backend should support an explicit SYCL build option.

Conceptually:

```text
-DGGML_SYCL=ON
```

or whatever option is appropriate for the vendored stable-diffusion.cpp / GGML revision.

The build system should detect the oneAPI compiler/toolchain rather than silently pretending SYCL is available.

Recommended configuration states:

```text
CPU build
CUDA build
SYCL build
multi-backend build when supported by the current GGML revision
```

If simultaneous backend linkage is not safe for the vendored revision, build separate native binaries and let Easy Diffusion choose the correct backend executable.

## sdkit3 command-line integration

Add backend selection without changing existing defaults.

Possible CLI:

```text
--backend auto
--backend cpu
--backend cuda
--backend sycl
```

Optional device selection:

```text
--device 0
--devices 0,1,2,3
```

The existing CUDA behavior should remain unchanged when SYCL is not requested.

## Configuration integration

Easy Diffusion should be able to express backend selection in its configuration file.

Example:

```yaml
backend: auto
sycl_devices: [0, 1, 2, 3]
sycl_multi_gpu: true
```

Possible advanced settings:

```yaml
sycl:
  enabled: true
  devices: [0, 1, 2, 3]
  strategy: auto
  cpu_fallback: true
  reserve_mb_per_device: 512
```

The exact schema can be adjusted to the existing Easy Diffusion configuration style.

## Model loading

Model loading must become backend-aware.

For SYCL:

1. Parse model metadata on CPU.
2. Memory-map model weights where possible.
3. Allocate device buffers only for tensors assigned to a SYCL device.
4. Avoid staging the entire model through an unnecessary second RAM copy.
5. Respect each device's maximum allocation size.
6. Allow layers to remain CPU-resident when they do not fit or are not worth transferring.

The goal is to make the existing mmap/offload infrastructure work with SYCL instead of creating a parallel loading system.

## UNet and DiT execution

The main diffusion network is the highest-priority inference target.

The desired execution path is:

```text
prompt conditioning
      ↓
latent input
      ↓
UNet / DiT graph
      ↓
GGML SYCL backend
      ↓
Intel GPU
```

Priority operations include:

- matrix multiplication
- convolutions where applicable
- normalization
- activation functions
- attention
- tensor reshape/permutation
- elementwise operations

Before adding custom SYCL kernels, verify whether the vendored `ggml-sycl` backend already implements the operation adequately.

## CLIP and text encoders

Text encoders should be independently placeable.

Example policy:

```text
UNet / DiT -> SYCL GPU 0
CLIP       -> SYCL GPU 1
VAE        -> SYCL GPU 2
ControlNet -> SYCL GPU 3
```

or, when memory is constrained:

```text
UNet / DiT -> SYCL GPU 0
CLIP       -> CPU
VAE        -> CPU
ControlNet -> SYCL GPU 1
```

This fits the existing Easy Diffusion approach of selectively putting subsystems on CPU or GPU.

## VAE encode and decode

VAE should support three modes:

```text
CPU
single SYCL GPU
SYCL with tiling
```

Existing VAE tiling should remain backend-neutral.

The tiling scheduler should create tiles on the host and dispatch the same graph to the selected SYCL device rather than implementing a completely separate VAE path.

For multi-device systems, future work may distribute independent VAE tiles across GPUs.

## ControlNet

ControlNet execution should use the same backend abstraction as the main diffusion network.

Preferred behavior:

```text
ControlNet device = auto
```

with optional override:

```text
--control-net-device sycl:1
```

This allows the main network and ControlNet to occupy different XG310 devices when memory pressure makes that useful.

## LoRA

LoRA support has two separate concerns:

1. Native LoRA file conversion / loading.
2. LoRA application during Easy Diffusion inference.

The runtime application path should not care whether the compute device is CUDA or SYCL.

The ideal model is:

```text
LoRA metadata / tensor mapping
            ↓
backend-neutral graph operation
            ↓
GGML backend
        ├── CUDA
        ├── SYCL
        └── CPU
```

Avoid a SYCL-specific LoRA implementation unless a backend operation is genuinely missing.

## Attention

Easy Diffusion currently has multiple attention-related modes and optimizations.

SYCL support should be layered:

### Level 1

Use generic GGML attention operations on SYCL.

### Level 2

Use any existing memory-efficient attention implementation already available in the vendored SYCL backend.

### Level 3

Add Intel-specific optimized attention kernels only after correctness and profiling show that generic SYCL is insufficient.

CUDA-only flags must not silently select CUDA code on a SYCL backend.

A backend capability structure is preferable:

```cpp
struct BackendCapabilities {
    bool flash_attention = false;
    bool fused_attention = false;
    bool fp16 = false;
    bool bf16 = false;
    bool int8 = false;
};
```

Then Easy Diffusion can enable only supported optimizations.

## VRAM budgeting

The existing `--max-vram` concept should be generalized so it can describe SYCL devices.

Possible syntax:

```text
--max-vram sycl0=7.0,sycl1=7.0,sycl2=7.0,sycl3=7.0
```

or configuration:

```yaml
vram_budget:
  sycl:0: 7.0
  sycl:1: 7.0
  sycl:2: 7.0
  sycl:3: 7.0
```

Always leave a configurable safety reserve instead of allocating every reported byte.

## Multi-GPU strategy

Because the XG310 exposes multiple independent GPUs, multi-GPU support should be explicit.

Three useful strategies are recommended.

### Strategy A: component placement

Place major model components on separate GPUs.

```text
GPU 0 -> UNet / DiT
GPU 1 -> CLIP / text encoder
GPU 2 -> VAE
GPU 3 -> ControlNet / auxiliary model
```

This is the safest first multi-GPU implementation because synchronization occurs at coarse boundaries.

### Strategy B: layer partitioning

Split model layers across GPUs.

```text
GPU 0 -> blocks 0-7
GPU 1 -> blocks 8-15
GPU 2 -> blocks 16-23
GPU 3 -> blocks 24-31
```

This requires explicit transfer of intermediate activations between devices.

It should come later than component placement because PCIe transfer overhead may dominate.

### Strategy C: independent job scheduling

Run separate image jobs on separate GPUs.

```text
request 1 -> GPU 0
request 2 -> GPU 1
request 3 -> GPU 2
request 4 -> GPU 3
```

For batch or queue workloads, this may provide better aggregate throughput than splitting one image across four GPUs.

## Scheduler abstraction

Suggested interface:

```cpp
enum class MultiGpuStrategy {
    Disabled,
    Components,
    Layers,
    Jobs,
    Auto,
};

struct DeviceAssignment {
    int unet = -1;
    int text_encoder = -1;
    int vae = -1;
    int controlnet = -1;
};
```

The scheduler should use measured free memory and model component size rather than only round-robin assignment.

## Tensor streaming

Easy Diffusion already has layer-streaming concepts. SYCL should participate in the same mechanism.

Desired flow:

```text
mmap model tensor
      ↓
pinned/shared host staging when useful
      ↓
copy required layer to SYCL device
      ↓
execute
      ↓
release or reuse device allocation
```

Do not require all weights to remain resident when the model exceeds the memory of one device.

## USM and memory policy

SYCL USM may be useful, but shared USM should not automatically replace explicit device allocations everywhere.

Recommended policy:

- device USM for hot compute tensors when supported well
- host/pinned allocations for transfer staging
- shared USM only where profiling shows it is appropriate
- mmap for model files
- explicit synchronization at graph/component boundaries

The implementation should remain compatible with backends where USM behavior differs.

## CPU fallback

SYCL support must not make unsupported operations fatal unless strict mode is requested.

Preferred graph behavior:

```text
operation supported by SYCL?
       |
      yes -> SYCL
       |
      no  -> CPU fallback
```

Add a strict diagnostic mode for development:

```text
--sycl-strict
```

In strict mode, unsupported fallback should fail and identify the missing operation so backend coverage can be improved.

## Backend capability reporting

Add a diagnostic command such as:

```text
sdkit --list-devices
```

and/or:

```text
sdkit --backend-info
```

Suggested output:

```text
Backend: SYCL
Device 0: Intel Server GPU
Memory: 8192 MiB
FP16: yes
BF16: yes/no
Flash attention: yes/no
VAE: supported
ControlNet: supported
```

This is important because oneAPI installation success does not guarantee that every model operation is accelerated.

## Easy Diffusion UI

Eventually expose a small hardware section in the UI:

```text
Compute backend: Auto / CUDA / SYCL / CPU
Primary device: Auto / SYCL 0 / SYCL 1 / ...
Multi-GPU: Off / Auto / Components / Jobs
```

Advanced users can receive per-component placement controls later.

The initial UI should remain simple and use automatic placement by default.

## Compatibility rules

1. Existing CUDA behavior must not regress.
2. CPU-only builds must remain valid.
3. SYCL must be optional at build time.
4. Runtime backend selection must not assume Intel hardware exists.
5. Unsupported SYCL operations should fall back to CPU unless strict mode is enabled.
6. Model files and GGUF output should remain backend-independent.
7. Backend-specific code should remain below the Easy Diffusion model/task layer whenever possible.

## Recommended implementation phases

### Phase 1 - Build and discovery

- build stable-diffusion.cpp with `ggml-sycl`
- enumerate Intel GPUs
- add `--backend sycl`
- add `--list-devices`
- run a minimal GGML operation on SYCL

### Phase 2 - Single-GPU image generation

- model load on SYCL
- text encoder on SYCL
- UNet / DiT on SYCL
- VAE decode on SYCL
- generate one image correctly
- compare against CPU output within expected numeric tolerance

### Phase 3 - Feature parity

- LoRA
- ControlNet
- VAE tiling
- mmap
- offload-to-CPU
- max-VRAM budgeting
- layer streaming
- image CLIP / text encoder placement

### Phase 4 - Performance

- profile attention
- profile host/device transfer
- reduce synchronization
- reuse allocations
- add Intel-specific kernels only where justified

### Phase 5 - XG310 multi-GPU

- enumerate all board GPUs
- component placement
- per-device VRAM budgets
- independent job scheduling
- optional layer partitioning
- automatic scheduling heuristics

## Relationship to native model conversion

The native Hugging Face / LoRA conversion plan and the Easy Diffusion oneAPI plan should share infrastructure where useful, especially:

- SafeTensors parsing
- mmap
- device enumeration
- backend capability reporting
- tensor dtype conversion
- quantization helpers

However, model conversion must remain functional on CPU and must not require oneAPI.

Likewise, Easy Diffusion inference should be able to use oneAPI even when conversion was performed elsewhere.

## Target architecture

The intended long-term structure is:

```text
Easy Diffusion UI / API
        |
        v
sdkit3 task/model layer
        |
        +---------------------------+
        |                           |
        v                           v
stable-diffusion.cpp          conversion tools
        |                           |
        v                           v
      GGML                  native tensor utilities
        |
        +---------+---------+---------+
        |         |         |         |
       CPU       CUDA      SYCL     other GGML backends
                           |
                           v
                      oneAPI / Intel GPU
```

The key principle is that **oneAPI should become a first-class Easy Diffusion compute backend, not a special-case conversion accelerator**.
