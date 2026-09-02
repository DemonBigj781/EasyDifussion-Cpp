# Split Attention feature

Split Attention is a separate attention feature beside xFormers.

The current method boundary is derived from the existing planning and end-to-end
prototype plus the stages required to preserve exact attention semantics across
chunks:

- `plan.cpp`
- `split.cpp`
- `qkt.cpp`
- `mask.cpp`
- `softmax.cpp`
- `av.cpp`
- `merge.cpp`
- `forward.cpp`

These Common files are structural placeholders, not claims of runtime support.
The prototype currently exposes only `make_plan` and `forward`; its Q * K^T,
masking, normalization, value accumulation, and online merge operations remain
internal stages. A backend may fuse multiple logical methods into one native
kernel while its definition and translation still document equivalent behavior.

Planned architecture follows the same translate-first, unify-second model:

- backend-native contracts live under `[backend]/definition/[device-type]/`
- backend-to-Common adapters live under `[backend]/translation/[device-type]/`
- unified cross-backend code will live directly under `common/`
- application code should ultimately call only the common feature API

The overlap with xFormers (`qkt`, `mask`, `softmax`, `av`, and `forward`) is
semantic. Split-specific `plan`, `split`, and `merge` methods remain separate
because chunk selection and exact cross-chunk reduction are part of this
feature's behavior.
