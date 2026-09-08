# oneAPI backend gap audit

Date: 2026-09-08  
Theory commit inspected: `9764a798b30b1412810030160ec5888bc0e59f93`  
Branches compared: `origin/main`, `origin/Theory`

## Verdict

Neither Main nor Theory currently has complete oneAPI support.

- Main contains a oneAPI/SYCL plan, toolchain and workflow scaffolding, and the
  vendored stable-diffusion.cpp GGML-SYCL backend. Those pieces do not expose a
  usable Easy Diffusion oneAPI backend.
- Theory adds a Common-routed oneAPI device-detection path and a library
  handler. It does not add oneAPI load/unload, inference execution, attention,
  memory placement, or end-to-end generation.
- Workflow 038 can configure/build the vendored `SD_SYCL` path, but a successful
  compile is build evidence only. It must not be reported as application or
  runtime support.

The current honest status is **detection foundation plus native backend build
scaffolding; generation support not implemented or validated**.

## Required architecture

Every DiffUser-owned oneAPI operation must follow the same contract as every
other backend:

`oneAPI definition -> translation -> Common -> translation -> oneAPI definition`

Native SYCL types, queues, events, USM pointers, Level Zero handles, kernels,
and driver errors belong in oneAPI definitions. Translations only convert
between those native values and Common messages. INFERENCE and the application
may call Common, never oneAPI definitions or translations directly.

The vendored GGML-SYCL engine remains an independent library implementation.
Using it does not permit DiffUser or INFERENCE to bypass Common when selecting,
loading, placing, or invoking backend-owned resources.

## Branch accounting

| Capability | Main | Theory | Required next proof |
| --- | --- | --- | --- |
| Toolchain provisioning | Scaffolded | Scaffolded | Build image succeeds |
| GGML-SYCL source | Vendored | Vendored | Required ops compile |
| Easy Diffusion SYCL configure | Scaffolded | Scaffolded | Workflow 038 complete compile |
| Device detection | No DiffUser route | Common-routed definition/translation | Runtime enumeration |
| Stable device identity | Absent | Absent | IDs remain unambiguous across CPU/GPU/NPU lists |
| Per-device memory limits | Absent | Total memory only | Global memory, max allocation, free/budget reporting |
| Model load/unload | Absent | Absent | Common lifecycle round trip |
| Attention | Absent | Empty oneAPI layout/inventory only | Generic GGML attention correctness first |
| INFERENCE execution | Absent | Absent | CLIP, denoiser, and VAE stage proofs |
| Full image generation | Absent | Absent | Reference-comparable output |
| Multi-device XG310 | Absent | Absent | Independent-device scheduling and measured transfers |

## XG310 rules

The H3C XG310 32 GB board must initially be treated as multiple independent
Intel GPU devices. Aggregate board memory must not be advertised as one
allocatable 32 GB pool. Each enumerated device needs its own stable selector,
memory budget, maximum-allocation limit, queue/context ownership, and failure
boundary. Peer-to-peer access and shared allocation are unsupported until the
installed board and driver prove otherwise.

The existing Theory detector numbers matches independently inside its CPU,
GPU, and NPU routes. Before those results can select execution devices, identity
must be strengthened so a selector cannot become ambiguous when the three
lists are combined. Platform, vendor/device identity, and preferably PCI/Level
Zero identity should be preserved by the native definition and normalized by
translation.

## Hardware safety gate

The passively cooled XG310 must not be placed under inference load until forced
airflow is installed and verified. The ordered T10 fan may be mechanically
adapted for that purpose. Software-only inspection, container/toolchain builds,
and CI compile tests can proceed before the fan arrives; on-card allocation,
kernel, generation, stress, and thermal tests cannot.

## Smallest correct implementation sequence

1. Strengthen oneAPI detection and its Common result with stable identity,
   per-device allocation limits, and capabilities needed for scheduling.
2. Add a public Common-only detection test that rejects a falsely unified
   XG310 memory pool and preserves independent device records.
3. Complete workflow 038 against the vendored stable-diffusion.cpp SYCL target.
4. Add oneAPI model lifecycle definitions and translations using the existing
   typed Common registry. Do not reuse CUDA-native request fields.
5. Connect one selected SYCL GPU to INFERENCE stage execution through Common;
   validate CLIP, VAE, then UNet/DiT independently.
6. Produce one complete single-device generation before implementing XG310
   multi-device placement.
7. Add multi-device scheduling only after per-device memory and host-staged
   transfer behavior have been measured on the cooled board.

Attention variants, overflow, and peer-to-peer optimization are later work.
The first generation should use the generic operations already supplied by the
vendored GGML-SYCL backend wherever they are correct.

## Promotion rule

oneAPI may be promoted from Theory to Main only when the promoted slice has a
Common-routed public contract test and does not regress existing CPU/CUDA
routes. “Toolchain installed,” “SYCL enabled,” “device detected,” and “compiled”
must remain separate status labels from “component ran” and “image generated.”

