# Existing application model/graph boundary

## Ownership decision

The active application already has model architecture and lifetime machinery:
ImageGenerator owns its sd_ctx; ModelLoader interprets checkpoint formats/names;
ModelManager owns parameter residency and staging; GGMLRunner builds and executes
graphs through those existing managers. The Theory generic GGUF owner must not
replace these owners or lend raw tensor handles to them.

The contract fixture therefore uses the real application ModelLoader,
ModelManager, GGMLRunner and Linear implementation. It does not add a second
linear layer, scheduler or model abstraction. The generic Theory resource is
released before the application's independent loader/runner starts.

Theory's separate INFERENCE tree remains planning/partial host-policy work.
This fixture validates the active application boundary before any such migration.

## Compatibility corrections

The bundled application defines GGML_MAX_NAME=160. The optional resource target
now inherits that exact directory definition instead of compiling against the
header's shorter default. It propagates the definition to consumers using GGML
headers, keeping the application/library tensor layout consistent.

Generic GGUF TensorInfo.shape is explicitly a serialized storage shape. The
application's existing GGUF reader can restore a different logical shape using
comfy.gguf.orig_shape metadata. A quantized convolution fixture proves this
distinction: stored [256,45] versus logical GGML [3,3,4,320]. Application graph
construction must use the existing application loader's interpretation.

## Realistic component fixture

The generated F32 checkpoint contains CLIP text-encoder FC1 names and dimensions:
768 input features, 3,072 output features, bias and two input tokens.
Deterministic bounded weights allow every output to be compared against an
independent scalar reference. It is not a pretrained CLIP checkpoint.

The fixture exercises three forward passes in each of Disk and ParamBackend
residency modes, eager loading followed by removal of the source file, explicit
unregistration before runner context destruction, and rejection of incompatible
parameter dimensions. It also checks long tensor names and original-shape
metadata using the actual application's reader.

## Scope and next boundary

SDKIT_BUILD_THEORY_APPLICATION_CONTRACTS is opt-in and OFF by default, alongside
the already optional resource target. Existing UIs, inpainting and generation
paths remain untouched.

A successful component test is not full CLIP, diffusion, image generation or
hardware-backend parity. It does not test pretrained weights, attention/GELU,
tokenizers, the complete text encoder, video or GPU execution. It establishes
which current owners and graph code must be reused rather than replaced.

## Validation on 2026-11-10

- CPU Release: full existing stable-diffusion shared library built; all six
  optional Theory CTest contracts passed, including numerical application graph.
- CPU Debug: full stable-diffusion library and component graph built with
  AddressSanitizer and UndefinedBehaviorSanitizer; application graph passed.
  LeakSanitizer is disabled because the executor's ptrace environment prevents it.
- Five standalone host lifecycle contracts, 117 foundation tests and 21 existing
  image/SAM/inpainting contract tests passed. These image tests are adapter/unit
  regressions, not model inference.
- CUDA and Vulkan were OFF; no GPU enumeration or pretrained model runs occurred.
