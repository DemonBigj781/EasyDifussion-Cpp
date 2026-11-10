# Flex Attention feature

Flex Attention is tracked as a resource-side attention selection feature. It
does not own SDKIT3 graph construction, inference orchestration, or sampling.

## Current CPU source

The CPU/host implementation now follows the normalized route:

```text
Common -> cpu/translation/cpu -> cpu/definition/cpu
```

Common owns request/result validation, backend dispatch, and compatibility
overloads. The CPU translation converts between Common and native values. The
CPU definition owns the high-resolution token-selection algorithm derived from
these logical stages:

- `mask_size`
- `validate`
- `sum_heads`
- `margin`
- `normalize`
- `threshold`
- `pool`
- `expand`
- `select`

The CPU translation is registered with Common in source. The public Common-only
`source/API.test/Feature/Attention/Flex/Cpu/Main.cpp` target validates translated
capabilities, deterministic 2x2-to-4x4 selection and expansion, compatibility
wrappers, insufficient output capacity, and rejection of an unregistered
backend. Run it from `source/API.test` with `make run-flex-cpu`.

See `native-calls/cpu.md` for the present inventory.
