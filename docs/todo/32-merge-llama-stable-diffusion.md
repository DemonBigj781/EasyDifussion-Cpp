# 32 — Merge llama.cpp into stable-diffusion.cpp Without GGML ABI / Symbol Collisions

## Objective
Allow llama.cpp functionality and stable-diffusion.cpp functionality to coexist in one native process/library architecture without conflicting GGML revisions, duplicate symbols, or incompatible ABI assumptions.

## Core problem
Both projects vendor/use GGML but may pin different revisions and compile overlapping symbols. Simply linking both static libraries can cause duplicate symbols or, worse, compile successfully while one component calls structures/functions from an incompatible GGML ABI.

## Implementation options
1. **Preferred long-term: shared GGML baseline.** Identify required GGML features/patches in each project, choose one compatible GGML revision, port both codebases to it, and build GGML once.
2. **Transitional: symbol namespace isolation.** Build one dependency with renamed/prefixed GGML symbols and fully isolated headers/types. This is maintenance-heavy and should not be mistaken for ABI unification.
3. **Current safe fallback: process isolation.** Keep llama and diffusion as separate processes while native APIs are stabilized. This is preferable to an unsafe in-process merge.

## Detailed shared-baseline process
- Inventory exact GGML commits and local patches.
- Generate API/struct difference report for headers used by each project.
- Identify backend differences (CUDA/SYCL/ROCm, quantization, graph APIs, allocators).
- Port the smaller patch set onto the selected baseline.
- Build one GGML target and link both llama and diffusion code against it.
- Remove duplicate vendored GGML compilation from one side.
- Add compile-time/static assertions where structures/layouts matter.
- Run llama and diffusion workloads in the same process repeatedly to detect allocator/backend global-state conflicts.
- Define ownership/lifecycle for backend registries, GPU contexts, logging, thread pools, and model memory.

## Relation to native converters
A shared native GGUF/conversion layer should avoid importing another copy of GGML. Keep conversion metadata/tensor parsing modular.

## Validation
Linker symbol audit, ABI/header consistency, CPU generation + LLM inference, GPU backend initialization for both, alternating workloads, simultaneous contexts if supported, unload/reload, quantized models, and sanitizers where feasible.

## Complete when
One process can use both llama and diffusion functionality against a deliberately-defined GGML architecture with no duplicate symbols, no version-dependent undefined behavior, and regression coverage. Until then, process isolation remains the supported fallback.
