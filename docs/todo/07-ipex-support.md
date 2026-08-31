# 07 — IPEX Support

## Objective
Provide Intel Extension for PyTorch compatibility for transitional or external Python workloads without making IPEX part of the long-term native core.

## Implementation
1. Inventory features that still use PyTorch and determine whether IPEX materially improves them.
2. Keep IPEX behind a Python adapter with strict version compatibility checks for Python, PyTorch, and IPEX.
3. Detect Intel device support and expose IPEX only when the installed stack is compatible.
4. Avoid adding new core image-generation dependencies on IPEX if the native SYCL path can perform the task.
5. Add explicit settings identifying when a task is routed through IPEX versus native oneAPI/SYCL.
6. When a Python feature is replaced natively, remove its IPEX dependency instead of maintaining duplicate implementations indefinitely.

## Dependencies
Python 3.13 migration if the selected IPEX/PyTorch versions support it. If not, this objective may need to remain an optional compatibility environment rather than part of the main venv.

## Failure behavior
An incompatible IPEX stack should disable the IPEX option, not prevent Easy Diffusion from starting.

## Validation
Test import/version probing, Intel device selection, one representative workload, fallback to CPU/native backend, and coexistence with the main server.

## Complete when
IPEX can accelerate explicitly-supported remaining Python tasks without obstructing the native oneAPI strategy or Python-removal roadmap.
