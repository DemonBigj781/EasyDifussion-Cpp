# xFormers translation call normalization

## Purpose

The `translation/` folder is the normalization boundary between backend-native APIs and the xFormers Common layer.

The intended direction is:

```text
backend-native call / intrinsic / kernel semantic
        |
        v
[backend]/translation/[device]/...
        |
        v
one normalized xFormers call vocabulary
        |
        v
xformers/common
        |
        v
Library / application
```

Common must not need to know whether a normalized operation came from CUDA, HIP/ROCm, SYCL, OpenCL, CPU C++, or another backend.

The translation layer is therefore responsible for taking calls that are equivalent in meaning but different in spelling, argument model, execution model, or error model and exposing them as one normalized operation.

---

## Core rule

> **Definitions describe what a backend/native operation means. Translation converts that backend meaning into the normalized xFormers call. Common consumes only the normalized call.**

Example:

```text
CUDA: __syncthreads()
ROCm/HIP: __syncthreads()
SYCL: nd_item.barrier()
OpenCL: barrier(CLK_LOCAL_MEM_FENCE)
CPU: thread barrier / staged synchronization

                |
                v
translation layer
                |
                v
xformers::sync_workgroup()
```

Common sees only:

```text
sync_workgroup()
```

It does not contain backend-specific branches for the native spelling.

---

## Initial normalized call vocabulary

The first vocabulary should come directly from operations already required by the xFormers implementations. Do not add speculative calls.

| Normalized call | CUDA native form | HIP/ROCm native form | SYCL/oneAPI form | OpenCL form | CPU form |
| --- | --- | --- | --- | --- | --- |
| `local_thread_id_x()` | `threadIdx.x` | `threadIdx.x` | `nd_item.get_local_id(...)` | `get_local_id(0)` | worker/thread index |
| `workgroup_id_x()` | `blockIdx.x` | `blockIdx.x` | `nd_item.get_group(...)` | `get_group_id(0)` | task/chunk index |
| `local_size_x()` | `blockDim.x` | `blockDim.x` | `nd_item.get_local_range(...)` | `get_local_size(0)` | worker count |
| `sync_workgroup()` | `__syncthreads()` | `__syncthreads()` | work-group barrier | `barrier(...)` | thread barrier when parallel |
| `local_memory<T>()` | `__shared__` | `__shared__` | local accessor / work-group local memory | `__local` | shared scratch storage |
| `fp16_to_fp32()` | `__half2float()` | HIP half conversion | `sycl::half` conversion | `convert_float()` | software/native half conversion |
| `bf16_to_fp32()` | NVIDIA BF16 conversion/bit reconstruction | HIP BF16 conversion | oneAPI BF16 conversion | extension/backend-specific | software/native BF16 conversion |
| `bitcast_u32_to_f32()` | `__uint_as_float()` | HIP bit reinterpretation | `bit_cast` equivalent | `as_float()` | `std::bit_cast<float>` |
| `math_exp()` | `expf()` | `expf()` | `sycl::exp()` | `exp()` | `std::exp()` |
| `math_tanh()` | `tanhf()` | `tanhf()` | `sycl::tanh()` | `tanh()` | `std::tanh()` |
| `math_max()` | `fmaxf()` | `fmaxf()` | `sycl::fmax()` | `fmax()` | `std::fmax()` |
| `math_isnan()` | `isnan()` | `isnan()` | `sycl::isnan()` | `isnan()` | `std::isnan()` |
| `positive_infinity()` | `CUDART_INF_F` | HIP/device infinity | SYCL/device infinity | `INFINITY` | numeric limits |
| `execution_context()` | `cudaStream_t` | `hipStream_t` | `sycl::queue` | `cl_command_queue` | executor/context |
| `launch_1d()` | CUDA kernel launch | HIP kernel launch | `parallel_for(nd_range)` | `clEnqueueNDRangeKernel` | loop/task dispatch |
| `last_execution_error()` | `cudaGetLastError()` | `hipGetLastError()` | exception / async handler | `cl_int` / event status | error/exception |

These names are design vocabulary, not yet a frozen ABI.

---

## Example: synchronization

Backend-native forms:

```text
CUDA      -> __syncthreads()
HIP       -> __syncthreads()
SYCL      -> item.barrier(...)
OpenCL    -> barrier(CLK_LOCAL_MEM_FENCE)
```

Translations normalize those into the same semantic call:

```text
cuda/translation/gpu/...    -> sync_workgroup()
rocm/translation/gpu/...    -> sync_workgroup()
oneapi/translation/gpu/...  -> sync_workgroup()
opencl/translation/gpu/...  -> sync_workgroup()
```

Common is written against `sync_workgroup()` semantics only.

The translation implementation may be a wrapper, inline adapter, macro-free function, callable object, or backend-specific implementation detail. The important part is the direction of responsibility.

---

## Example: asynchronous execution

Native concepts are similar but not identical:

```text
CUDA   -> cudaStream_t
HIP    -> hipStream_t
SYCL   -> sycl::queue
OpenCL -> cl_command_queue
```

Translation normalizes them as an xFormers execution context rather than forcing Common to understand four queue/stream systems.

```text
native stream/queue
      |
      v
translation
      |
      v
normalized execution_context
```

Backend-native handles remain below translation. Common may pass an opaque normalized context/reference, but it must not inspect a `cudaStream_t`, `hipStream_t`, `sycl::queue`, or `cl_command_queue` directly.

---

## Example: xFormers forward execution

The normalized *calls* and the normalized *algorithm* are separate concerns.

CPU can use normalized calls to execute staged attention:

```text
qkt()
apply_mask()
softmax()
av()
```

CUDA may use a fused native implementation internally, but its translation still presents the same normalized xFormers semantic entry point:

```text
xformers_forward(request, context)
```

Therefore:

```text
Common
  -> xformers_forward(normalized request)
       |
       +-> CPU translation: staged implementation
       +-> CUDA translation: fused kernel
       +-> ROCm translation: future HIP/native implementation
       +-> oneAPI translation: future SYCL/native implementation
```

The Common call remains the same even when backend mechanics differ.

---

## What belongs in definition

Definition files document backend-native meaning and constraints, for example:

- native function/intrinsic/operator name;
- argument meaning;
- supported dtypes;
- synchronization requirements;
- memory/address-space requirements;
- architecture restrictions;
- error behavior;
- native return/result semantics;
- differences from other backend equivalents.

Definition answers:

```text
"What does this backend call do?"
```

---

## What belongs in translation

Translation implements the mapping:

```text
"How do I expose that backend behavior as the normalized xFormers call?"
```

Translation owns:

- renaming/native-call adaptation;
- argument reordering;
- type conversion;
- shape/layout conversion;
- stream/queue/context conversion;
- error conversion;
- small composed equivalents when one native call is insufficient;
- selection between multiple native calls that represent one normalized operation;
- backend-specific workarounds needed to preserve normalized semantics.

---

## What belongs in Common

Common owns the normalized semantic API and orchestration.

Common should be able to say things like:

```text
xformers_forward(request)
query_capability(...)
normalize_mask(...)
normalize_error(...)
```

without containing code like:

```text
if CUDA -> cudaFoo()
if ROCm -> hipFoo()
if SYCL -> queue.foo()
```

Those choices belong in translation.

---

## Required comparison documentation

For every normalized xFormers call, maintain a comparison row containing:

1. normalized operation name;
2. CUDA/NVIDIA native equivalent;
3. ROCm/AMD native equivalent;
4. oneAPI/SYCL equivalent;
5. OpenCL equivalent where applicable;
6. CPU equivalent;
7. semantic differences;
8. argument differences;
9. dtype/layout differences;
10. synchronization differences;
11. architecture limitations;
12. validation state.

Suggested validation states:

```text
unknown
specified
compile-only
runtime-smoke
conformance-tested
```

---

## Current xFormers direction

The current CUDA translation already contains many native CUDA operations that should be extracted conceptually into this normalized map. CPU already demonstrates the logical staged operation. ROCm and oneAPI translations are still largely placeholders, making this comparison work the correct prerequisite before implementing them.

The next development task should therefore be:

```text
for every native operation used by CUDA xFormers
    identify normalized semantic call
    document AMD/HIP equivalent
    document oneAPI/SYCL equivalent
    document CPU equivalent
    record meaningful semantic differences
then
    implement translation wrappers/adapters per backend
```

Only after those translations exist should Common depend on the normalized calls.
