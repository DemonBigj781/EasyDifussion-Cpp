# INFERENCE.cpp

INFERENCE.cpp is the independently owned inference-orchestration project being
split from the SDKIT3 design. SDKIT3 remains intact under
`source/sdkit3-port-source` as reference material; its production implementation
is not linked or copied into this project.

INFERENCE.cpp owns model-family sequencing, conditioning, tokenization,
denoising schedules, sampler policy, guidance, generation state, and result
assembly. It does not own hardware discovery, memory residency, model-resource
load/unload, backend tensors, attention implementations, cache implementations,
or driver calls. Those are DiffUser concerns.

The initial build intentionally contains only the independent request and
execution-plan layer. See [MIGRATION_MANIFEST.md](MIGRATION_MANIFEST.md) for the
SDKIT3 disposition map and the order in which inference behavior will be
reimplemented.

## Dependency rule

Production files under `include/` and `src/` may use only the C++ standard
library. They must not include or link SDKIT3, stable-diffusion.cpp, GGML,
llama.cpp, DiffUser.cpp, REST transport, or UI code. Composition belongs in a
separate integration target, currently API.test.

## Build

```sh
cmake -S source/INFERENCE.cpp -B build/inference
cmake --build build/inference
ctest --test-dir build/inference --output-on-failure
```
