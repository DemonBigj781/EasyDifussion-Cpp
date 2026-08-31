# EasyCache — CUDA

## Status

**Inventory pending**.

This page is reserved for CUDA-specific EasyCache behavior behind the API.cpp backend abstraction. The implementation status must be derived from the current code before this page is marked Native, Compatibility, Fallback, Stub, Unsupported, or Not implemented.

## Required audit

Document the CUDA integration point, supported model families, cache state placement, VRAM behavior, precision restrictions, compile/runtime requirements, invalidation rules, fallback behavior, and CI/runtime validation.

## Rule

Do not move generic EasyCache policy or model logic into the backend layer. Only CUDA-specific execution, memory, kernels, synchronization, and capability handling belong here.
