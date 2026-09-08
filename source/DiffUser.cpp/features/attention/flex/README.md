# Flex Attention feature

Flex Attention is tracked as a resource-side attention selection feature. It
does not own SDKIT3 graph construction, inference orchestration, or sampling.

## Current CPU source

The current CPU/host implementation lives in `common/flex_attention.hpp` and
`common/flex_attention.cpp`. It implements a monolithic high-resolution token
selection route derived from these logical stages:

- `mask_size`
- `validate`
- `sum_heads`
- `margin`
- `normalize`
- `threshold`
- `pool`
- `expand`
- `select`

This is implementation evidence, not a completed registered Common contract.
The source is self-contained under `edcpp::api::attention::flex`, but it is
still one host algorithm rather than an accommodated CPU route. Before CPU
support can advance beyond `I`, it must be separated into:

1. a backend-neutral Common request and result;
2. a CPU definition containing native host behavior;
3. a CPU translation registered with Common;
4. deterministic CPU validation for the logical selection semantics;
5. API.test coverage that exercises each advertised selection semantic.

See `native-calls/cpu.md` for the present inventory.
