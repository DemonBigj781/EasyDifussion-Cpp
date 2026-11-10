# GGUF CPU tensor-resource integration

## What now executes

The optional Theory resource target can load a real GGUF tensor container into
the existing GGML CPU backend, enumerate tensor metadata, and copy exact tensor
bytes back through GGML's tensor API. The public GgufCpuWeights owner is
move-only and exposes no raw GGML contexts, tensors or buffers.

The route remains facade -> normalized Common contract -> CPU translation ->
native CPU implementation. The native owner groups GGUF metadata, GGML tensor
context and backend buffer, releasing backend storage before the contexts.
Unsupported backends are rejected before opening a file or selecting a device.

This is model-weight resource loading, not model-family validation, graph
execution, SDKit generation or an image-quality claim. The test container has
two synthetic F32 tensors and a synthetic architecture label.

## Data movement and limits

1. Read one bounded file snapshot into host memory. This avoids reopening a
   potentially changed file separately for metadata and tensor payloads.
2. Check GGUF magic/version and tensor/KV-count limits before native parsing.
3. Parse metadata without allocating tensor data.
4. Validate each payload offset/size and total tensor bytes.
5. Query the CPU backend's aligned allocation requirement and enforce its budget.
6. Allocate through the existing GGML CPU buffer type, then use tensor_set to
   transfer each payload into its own tensor.
7. Release the file snapshot. Reads subsequently use tensor_get from the backend.

Defaults are 256 MiB file bytes, 256 MiB backend bytes, 4,096 tensors and 65,536
metadata entries. Callers may explicitly choose different limits. The budgets
are separate: peak memory also includes parser metadata and temporary host
readback copies. This is not a zero-copy/mmap or streaming loader. One readback
is bounded by the selected tensor size.

GGUF versions 2 and 3 are admitted by this wrapper. Model-family interpretation,
quantized operation execution, sharded files, device offload and other container
formats remain outside this increment.

## Lifetime

The CPU translation and GGML libraries are linked dependencies of the optional
target. No module is dlopened or dlclosed by this resource, and there is no
hot-unload route. Immutable concurrent reads are supported while the owner stays
alive. Racing a read with move/reset/destruction is prohibited.

## Verification

- Five CTest fixtures pass against the existing GGML CPU runtime.
- The GGUF fixture verifies exact full/partial bytes, shape/type/name metadata,
  16 load/move/release cycles, move assignment/self-move, repeated reset and
  256 reads across four threads.
- Removing the source file after loading does not affect backend readback.
- File/backend/count budgets, truncated payloads, invalid magic, unknown tensors,
  out-of-range accesses and unsupported backend routes are rejected.
- A fresh build instruments both the adapter and GGML dependencies with
  AddressSanitizer and UndefinedBehaviorSanitizer; all five CTest fixtures pass.
  LeakSanitizer is disabled because of the executor's ptrace incompatibility.
- Host lifecycle and foundation suites remain separate regression evidence.

SDKIT_BUILD_THEORY_CPU_RESOURCES remains OFF by default. Existing UI, SAM,
inpainting and model-generation paths are unchanged. No user weights, GPUs,
Media drives or video LoRAs were accessed.

## Next actual runtime boundary

An inference consumer must bind these owned tensors to a model architecture
and graph without retaining raw handles beyond owner lifetime. It must also
preserve scheduler/allocator conventions and prove forward-operation results.
This increment intentionally does not replace the established SDKit loader or
claim checkpoint compatibility from successful GGUF metadata parsing.
