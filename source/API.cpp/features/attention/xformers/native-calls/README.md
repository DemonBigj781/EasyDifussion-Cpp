# xFormers native-call inventories

This directory records backend-native execution primitives **before** any attempt is made to choose a shared translation vocabulary.

## Required order of work

1. Inventory the calls, intrinsics, execution objects, memory spaces, launch forms, dtype conversions, synchronization, and error/reporting mechanisms actually used or required by each backend.
2. Record backend-specific semantic constraints and architecture restrictions.
3. Compare inventories side by side.
4. Only then choose normalized calls for each backend `translation/` directory.
5. Common consumes normalized translation calls only; Common must not directly depend on CUDA, HIP, SYCL, OpenCL, or other vendor syntax.

## Rule

These files are descriptive inventories, not translation specifications. Similar-looking calls must not be declared equivalent here merely because their names resemble one another.

## Inventories

- `cuda.md` — NVIDIA CUDA primitives currently used by the xFormers CUDA path plus relevant nearby execution primitives.
- `rocm-hip.md` — AMD ROCm/HIP native primitives that may be needed by an equivalent implementation.
- `oneapi-sycl.md` — oneAPI/SYCL native primitives relevant to an equivalent implementation.
- `cpu.md` — host C++ operations used by the current CPU xFormers path.

The later comparison document should cite these inventories and explicitly state where behavior is exact, approximate, emulated, unsupported, architecture-dependent, or still unknown.
