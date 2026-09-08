# Split Attention CPU native-call inventory

## Status

Prototype-accounted only (`I`). The prototype is not a registered CPU backend
and the files in `common/` remain structural placeholders.

## Existing calls

| Source call | Present behavior | Planned logical methods | Current boundary |
| --- | --- | --- | --- |
| `prototype::make_plan` | Chooses unsplit, query-split, or K/V-split execution from shape and memory budget | `plan`, `split` | Exploratory CPU prototype |
| `prototype::forward` | Executes dense F32 attention according to a supplied plan | `qkt`, `mask`, `softmax`, `av`, `merge`, `forward` | Exploratory CPU reference |

The fused prototype stages are evidence for semantics only. They do not justify
marking the individual Common methods implemented or `N/A`.

## Unresolved before CPU Common support

- Freeze normalized tensor, mask, plan, slice, workspace, and result contracts.
- Move native CPU behavior under `cpu/definition/cpu/`.
- Add a CPU translation registered with Common.
- Preserve exact global softmax semantics across K/V chunks.
- Define grouped-query, broadcast-mask, stride, dtype, cancellation, and
  overflow behavior.
- Compare unsplit, query-split, and K/V-split results deterministically.
- Keep selection and sampling policy in SDKIT3; expose only the resource and
  attention operation required by the caller.
