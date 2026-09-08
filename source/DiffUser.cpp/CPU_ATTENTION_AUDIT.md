# CPU attention coverage audit

This audit prevents a directory, placeholder, prototype, or advertised
capability from being mistaken for usable CPU attention. It covers every
attention family currently present under `features/attention/`.

## Acceptance rule

A CPU family is accommodated only when it has a backend-neutral Common
contract, CPU-owned implementation, Common-to-CPU translation, repeatable
build target, and API.test runtime validation. Semantic forms are credited
only when the runtime test exercises them.

| Family | Common contract | CPU implementation | CPU translation | API.test build/runtime | CPU disposition |
| --- | --- | --- | --- | --- | --- |
| xFormers | yes | materialized CPU stages | yes | `run-xformers-cpu` passes | accommodated for the tested forms below |
| FlashAttention | yes | fused host forward currently lives in translation | yes | `run-flash-cpu` passes | usable baseline; ownership split and coverage remain incomplete |
| Flex Attention | selection algorithm only | no | no | no | not accommodated |
| Split Attention | placeholders; separate prototype | no | no | no | not accommodated |
| SageAttention | support registry only | no | no | no CPU test | not accommodated |

Flash is deliberately not called architecture-complete: its CPU translation
currently owns both adaptation and execution. A CPU definition must be
separated before it can reach `C` in the implementation ledger.

## Tested semantic forms

| Form | xFormers CPU | Flash CPU | Remaining requirement |
| --- | --- | --- | --- |
| self-attention | runtime tested | runtime tested | none for baseline F32 |
| cross-attention (different Q and K/V token counts) | runtime tested | runtime tested | none for baseline F32 |
| multi-head attention | runtime tested | capability only | add deterministic Flash case |
| grouped-query attention (GQA) | runtime tested | capability only | add deterministic Flash case |
| multi-query attention (MQA) | runtime tested | representable as GQA; not tested | add deterministic Flash case |
| additive/broadcast mask | runtime tested | capability only | add deterministic Flash case |
| causal mask | runtime tested | not in Flash request | define semantics or document exclusion |
| ALiBi/max-bias | runtime tested | capability only | add deterministic Flash case |
| logit soft-cap | runtime tested | capability only | add deterministic Flash case |
| attention sinks | explicitly rejected | not in Flash request | design and implement if required |
| F32 input/output | runtime tested | runtime tested | none for baseline |
| F16 input | explicitly rejected | implemented; not tested | add conversion/numerical test |
| BF16 input | explicitly rejected | implemented; not tested | add conversion/numerical test |
| strided tensors | accepted by implementation; not isolated in test | implemented; not tested | add padded/strided cases |
| batching | represented; not isolated in test | represented; not tested | add multi-batch cases |
| split/chunked exact attention | no | online fused execution, not Split API | implement Split family route |
| Flex token selection | no | no | implement Flex family route |
| Sage quantized attention | no | no | implement Sage CPU route or explicitly exclude it |

“Capability only” means the translation advertises and contains a path, but
the current deterministic CPU test does not prove that semantic form.

## Required order of work

1. Expand Flash CPU tests for every advertised capability, dtype, stride, head
   mapping, batch, and rejection case.
2. Move Flash host execution into `cpu/definition/cpu/`; leave registration and
   request adaptation in `cpu/translation/cpu/`.
3. Promote Flex from its Common-host selection prototype into a normalized
   contract, CPU definition, CPU translation, and deterministic API.test.
4. Promote Split's prototype into real Common methods and a CPU route, testing
   exact equivalence across query splits, K/V splits, masks, and merge order.
5. Decide whether Sage CPU is required. If it is, implement and test it; if it
   is not, record an explicit unsupported decision instead of leaving folders
   that imply support.

Audit evidence on 2026-09-08: `run-xformers-cpu` passed its 32 lifecycle cycles
and semantic/error suite; `run-flash-cpu` passed its current baseline numerical
and validation test. No CPU runtime target exists for Flex, Split, or Sage.
