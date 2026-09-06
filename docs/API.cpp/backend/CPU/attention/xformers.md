# CPU — xFormers attention

## Status
Implemented as the Common semantic/reference path.

## Current architecture
The CPU translation self-registers with the backend-neutral xFormers Common contract. Its `forward` callback performs the explicit `QKT -> mask -> softmax -> AV` stages and materializes a temporary F32 score buffer.

## Supported contract
The current path supports F32 Q/K/V/output, additive F32 masks with broadcasting, causal masking, explicit ALiBi slopes, logit soft-cap, GQA, MQA, and equal Q/K/V batch counts. F16, BF16, and attention sinks are rejected during Common validation.

This is a correctness-oriented execution plan, not a claim that CPU must reproduce the CUDA fused-kernel mechanics.

## Routing
Standalone tests call `edcpp::api::attention::xformers::forward()` and reach
the CPU translation only through Common registration. Any future exported
Library facade must call that same Common entry point.

## Validation
The exact `source/API.test/build/xformers-load-unload-cpu-test` program passes analytical output, masks, ALiBi, soft-cap, GQA/MQA, invalid-request cases, and 32 custom Common model load/forward/unload cycles.
