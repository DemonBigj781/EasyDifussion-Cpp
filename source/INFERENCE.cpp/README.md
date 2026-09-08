# INFERENCE.cpp

INFERENCE.cpp is the independently owned inference-orchestration project being
split from the SDKIT3 design. SDKIT3 remains intact under
`source/sdkit3-port-source` as reference material; its production implementation
is not linked or copied into this project.

INFERENCE.cpp owns model-family sequencing, conditioning, tokenization,
denoising schedules, sampler policy, guidance, generation state, and result
assembly. It does not own hardware discovery, memory residency, model-resource
load/unload, backend tensors, attention implementations, cache implementations,
or driver calls. Those are DiffUser concerns, requested only through DiffUser
Common contracts.

Feature implementations follow the DiffUser organizational convention under
`features/<feature>/common/`. Tokenization therefore begins under
`features/token/common/`; device definition and translation layers are added
only when a feature genuinely has device-specific behavior.

The initial build contains the independent request/execution-plan layer and a
generic BPE tokenizer foundation. The tokenizer is not yet CLIP- or T5-complete.
See [MIGRATION_MANIFEST.md](MIGRATION_MANIFEST.md) for the SDKIT3 disposition
map and remaining migration order.

## Dependency rule

Production files may use the C++ standard library and DiffUser Common headers.
They must not include a DiffUser definition, translation, backend handler, or
driver surface, and must not include or link SDKIT3, stable-diffusion.cpp, GGML,
llama.cpp, REST transport, or UI code. When inference requires a hardware or
resource operation not represented by Common, its normalized contract is added
to DiffUser Common and implemented through the applicable GPU definitions and
translations before INFERENCE.cpp consumes it. Raw backend buffers are never a
substitute for a missing Common resource contract.

The active request is text-input generation only. Image and mask inputs are
intentionally absent until DiffUser owns normalized image-resource contracts
and the required GPU translations.

## Build

```sh
cmake -S source/INFERENCE.cpp -B build/inference
cmake --build build/inference
ctest --test-dir build/inference --output-on-failure
```
