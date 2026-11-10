# 07 — IPEX Support

## Objective

Provide Intel Extension for PyTorch compatibility for retained or external
Python workloads while keeping the native compute core independent of IPEX.
Python remains required for APIs without practical C++ alternatives; follow the
[Python runtime policy](../python-runtime-policy.md).

## Implementation
1. Inventory features that still use PyTorch and determine whether IPEX materially improves them.
2. Keep IPEX behind a Python adapter with strict version compatibility checks for Python, PyTorch, and IPEX.
3. Detect Intel device support and expose IPEX only when the installed stack is compatible.
4. Avoid adding new core image-generation dependencies on IPEX if the native SYCL path can perform the task.
5. Add explicit settings identifying when a task is routed through IPEX versus native oneAPI/SYCL.
6. Remove an IPEX dependency only when its replacement is validated and no retained Python workload in that environment uses it. Native coverage of one feature does not retire other Python APIs.

## Dependencies
Python 3.13 migration if the selected IPEX/PyTorch versions support it. If not,
keep the optional compatibility stack in a separate uv-managed worker
environment with a pinned interpreter and its own lockfile. Use the same
locked/frozen deployment conventions as the main application environment.

## Failure behavior
An incompatible IPEX stack should disable the IPEX option, not prevent Easy Diffusion from starting.

## Validation
Test import/version probing, Intel device selection, one representative workload, fallback to CPU/native backend, and coexistence with the main server.

## Complete when

IPEX can accelerate explicitly supported Python tasks while coexisting with the
native oneAPI strategy and the maintained Python environments. Complete Python
removal is not a project objective or an acceptance criterion.
