# xFormers feature layout

Feature-first namespace for xFormers-compatible attention methods.

Canonical backend path: `API.cpp/features/attention/xformers/[backend]/definition/[device]/[method].cpp`.
Backend-neutral contracts remain in `API.cpp/features/attention/xformers/common/[method].cpp`.
Backend adapters live in `API.cpp/features/attention/xformers/[backend]/translation/[device]/[method].cpp`.

Each backend directory provides the same method names where that backend can support them. Each method file is initially a definition placeholder only; it documents the intended stage and must not be treated as a validated implementation until backend code and tests exist.

Initial method vocabulary:
- `qkt.cpp`: tiled/blockwise Q * K^T score generation.
- `mask.cpp`: attention mask/bias application to scores.
- `softmax.cpp`: numerically stable attention-score normalization.
- `av.cpp`: normalized attention weights applied to V.
- `forward.cpp`: complete memory-efficient forward path/orchestration.

Backend-specific implementation details remain isolated inside the backend directory. Existing API and attention implementations outside this new feature tree are not to be modified as part of establishing this structure.
