# INFERENCE.cpp

INFERENCE.cpp is the independently owned inference-orchestration project being
split from the SDKIT3 design. SDKIT3 remains intact under
`source/sdkit3-port-source` as reference material; its production implementation
is not linked or copied into this project.

INFERENCE.cpp owns model-family sequencing, conditioning, tokenization,
denoising schedules, sampler policy, guidance, generation state, result
assembly, and delivery of completed image/video results to its caller. It does
not own hardware discovery, memory residency, model-resource
load/unload, backend tensors, attention implementations, cache implementations,
or driver calls. Those are DiffUser concerns, requested only through DiffUser
Common contracts.

Tokenization follows the shared `features/token/common/` organization. VAE has
the explicit implementation layout
`Feature/vae/[backend]/[device]/[function].cpp`, with its normalized planning
contract under `Feature/vae/common/`. Backend/device function files are added
only when a real implementation exists.

The initial build contains the independent request/execution-plan layer and a
generic BPE tokenizer foundation. The tokenizer now also owns CLIP byte-to-Unicode
encoding, CLIP-compatible token splitting, and construction of the canonical
CLIP vocabulary from serialized merge rules. T5-style Unigram metaspace
pretokenization and scored segmentation are also independent and accept an
injected model vocabulary. A standard-library loader extracts that vocabulary,
its scores, special-token IDs, and metaspace policy from caller-provided
tokenizer JSON. The tokenizer resource boundary constructs CLIP or T5/UMT5
tokenizers from caller-owned byte views and reports their special-token policy;
it performs no filesystem or network access. Bundled production vocabulary
resources and checked-in family-specific golden vectors remain. The VAE feature
owns validated encode/decode shape and tiling plans but not yet execution. See
[MIGRATION_MANIFEST.md](MIGRATION_MANIFEST.md) for the SDKIT3 disposition map
and remaining migration order.

The host-only conditioning foundation under `features/conditioning/common/`
parses nested prompt attention, explicit weights, escapes, and `BREAK`
boundaries. Requests select natural-language or tag interpretation. Natural
language preserves prose; tag interpretation treats colon-linked fields such
as `red:fur` as one fusion group and emits `AND` clauses as distinct
conditioning branches. Both modes accept a final numeric field such as
`fur:1.2` and compound each trailing `+` by 1.1 (`fur++` is 1.21). Numeric
prose such as `16:9` remains literal. Parentheses retain their usual 1.1
multiplier and `BREAK` retains its fixed-chunk boundary behavior.

The tokenizer-neutral sequence policy attaches weights and per-token fusion
group IDs, records `AND` branch offsets, adds one declared special-token
envelope per chunk, pads or truncates, expands overflow into fixed chunks, and
emits binary or additive attention masks. It does not construct or combine
embeddings and does not call a backend; that execution remains gated on a
DiffUser Common text-conditioning resource contract.

## Dependency rule

Production files may use the C++ standard library and DiffUser Common headers.
They must not include a DiffUser definition, translation, backend handler, or
driver surface, and must not include or link SDKIT3, stable-diffusion.cpp, GGML,
llama.cpp, REST transport, or UI code. When inference requires a hardware or
resource operation not represented by Common, its normalized contract is added
to DiffUser Common and implemented through the applicable GPU definitions and
translations before INFERENCE.cpp consumes it. Raw backend buffers are never a
substitute for a missing Common resource contract.

The formal handoff is “DiffUser prepares; INFERENCE uses.” INFERENCE receives
only Common-prepared resources and operations, applies inference-owned model
logic to them, and delivers the completed result without taking over backend
allocation, placement, transfer, or cleanup.

The active request is text-input generation only. Image and mask inputs are
intentionally absent until DiffUser owns normalized image-resource contracts
and the required GPU translations.

## Build

```sh
cmake -S source/INFERENCE.cpp -B build/inference
cmake --build build/inference
ctest --test-dir build/inference --output-on-failure
```

External authoritative tokenizer fixtures can be checked without adding their
vocabularies to this repository. Each non-comment TSV line contains
comma-separated expected token IDs, a tab, and the literal prompt:

```sh
build/inference/edcpp_inference_tokenizer_golden_runner \
  t5-json /path/to/tokenizer.json /path/to/vectors.tsv
```

Use `clip-merges` in place of `t5-json` for a CLIP merge resource. CTest
smoke-tests both runner modes with tiny synthetic fixtures; authoritative
resources and vectors remain caller-supplied and are never accessed by the
default test suite.
