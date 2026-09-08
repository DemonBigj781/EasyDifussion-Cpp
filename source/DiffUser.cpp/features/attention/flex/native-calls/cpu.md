# Flex Attention CPU native-call inventory

## Status

Source-accounted only (`I`). No CPU definition/translation/Common roundtrip is
established yet.

## Existing calls

| Source call | Present behavior | Planned logical methods | Current boundary |
| --- | --- | --- | --- |
| `sd_flex_attention_mask_size` | Computes output-mask storage size | `mask_size` | SDKIT3-facing wrapper |
| `sd_flex_attention_select` | Converts public parameters and copies the selected mask | `select` | SDKIT3-facing wrapper |
| `valid_config` | Validates grid sizes and threshold | `validate` | Header implementation |
| `select_high_resolution_tokens` | Sums heads, removes padding margins, normalizes, thresholds, pools, and expands | `sum_heads`, `margin`, `normalize`, `threshold`, `pool`, `expand`, `select` | Monolithic host implementation |

## Unresolved before CPU Common support

- Define normalized tensor layout, ownership, and size rules.
- Decide which logical stages remain fused versus separately callable.
- Remove the dependency on `stable-diffusion.h` from the API implementation.
- Add CPU definition and translation directories and registration.
- Establish edge behavior for non-finite inputs, degenerate ranges, grid
  conversion, and output-buffer sizing.
- Add deterministic numerical and boundary validation.
