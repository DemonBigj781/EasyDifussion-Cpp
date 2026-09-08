# INFERENCE.cpp to DiffUser Common gap audit

Audited on 2026-09-08 against the first INFERENCE.cpp execution-plan and token
feature. This is a moving gate: it must be updated before each SDKIT3 inference
area is reimplemented in INFERENCE.cpp.

## Admission rule

INFERENCE.cpp may call a hardware-backed or resource-owning operation only when
DiffUser provides all of the following for the backend being claimed:

1. normalized Common request/result and capability contract;
2. backend definition;
3. Common-to-backend translation;
4. registration or dispatch through Common;
5. API.test build and runtime evidence.

A header, operation enum, placeholder directory, or implementation without a
translation does not pass this gate. Host-only inference policy, such as text
pretokenization and BPE merge selection, stays in INFERENCE.cpp and requires no
DiffUser operation.

Ownership test: if an operation prepares resources or primitive capability for
work, it belongs to DiffUser. If it consumes that prepared work to execute
model logic or deliver the finished result, it belongs to INFERENCE.cpp.

## Current operation gaps

| Inference need | Common contract | CPU route | GPU translation | Current decision |
| --- | --- | --- | --- | --- |
| model byte resource load/unload | yes | runtime tested | CUDA runtime tested; other GPU backends absent | usable only for proven backends |
| tokenization | not required | INFERENCE host implementation | not required | keep in `features/token/common` |
| text-conditioning tensor/resource | no | no | no | block conditioning migration |
| latent creation and ownership | no | no | no | block latent initialization execution |
| deterministic noise fill | no | no | no | add before sampler execution |
| model-family forward execution | INFERENCE-owned; not a DiffUser operation | required primitives incomplete | required GPU primitive translations incomplete | implementation blocked, ownership retained by INFERENCE |
| generic tensor operations needed by schedulers | only operation enum fragments | no complete route | no complete route | audit each scheduler before migration |
| VAE encode semantics | INFERENCE plan exists; not a DiffUser operation | handler resources absent | handler GPU translations absent | execution blocked |
| VAE decode semantics | INFERENCE plan exists; not a DiffUser operation | handler resources absent | handler GPU translations absent | execution blocked |
| normalized image resource | no | no | no | raw image pointers forbidden |
| normalized mask resource | no | no | no | raw mask pointers forbidden |
| device-to-host result transfer | no | no | no | add administrative Common transfer handler |
| result assembly and delivery | INFERENCE-owned; not a DiffUser operation | not applicable | not applicable | implement after transfer handler exists |
| xFormers CPU attention | yes | runtime tested | CUDA route exists | usable only through its Common contract |
| FlashAttention | yes | CPU baseline runtime tested | CUDA runtime tested | semantic test gaps remain |
| Flex Attention | partial algorithm, no registered CPU contract | no | no | blocked |
| Split Attention | prototype only | no | no | blocked |
| SageAttention | support contract only | no | CUDA forward contract incomplete | blocked for inference forward |

## First consequence

The active INFERENCE.cpp plan can validate text-generation intent but cannot
execute generation. `GenerationRequest::model` now refers to the existing
DiffUser Common model resource instead of inventing an INFERENCE-owned model
handle. Image and mask inputs were removed from the active request until their
Common resource contracts and applicable GPU translations exist.

## Required review for every added inference feature

1. List every external operation the proposed feature performs.
2. Separate host inference policy from hardware/resource operations.
3. Match each hardware/resource operation to an existing DiffUser Common call.
4. Add missing Common contracts to DiffUser before adding the inference call.
5. Add and test CPU plus every claimed GPU definition/translation.
6. Only then enable the feature in an INFERENCE.cpp execution plan.
