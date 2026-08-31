# 14 — Windows Support

## Objective
Provide a native Windows EasyDifussion-Cpp build and runtime without WSL or Wine.

## Implementation
1. Audit POSIX dependencies: signals, `fork`/process handling, paths, executable permissions, mmap, sockets, symlinks, shell scripts, environment variables, and filesystem semantics.
2. Extend existing `_WIN32` code in sdkit rather than adding duplicate platform layers.
3. Add CMake presets for MSVC and/or clang-cl and define supported compiler versions.
4. Provide Windows dependency acquisition and DLL deployment so users do not manually construct PATH.
5. Replace shell-only installer/start operations with cross-platform helpers or PowerShell equivalents.
6. Validate spaces/unicode in model paths and Windows file-lock behavior during model reload/conversion.
7. Handle Ctrl-C/service shutdown and child-process cleanup correctly.
8. Test GPU backends independently: CUDA first, then other Windows-capable backends.

## Dependencies
Cross-platform filesystem/process abstractions should be centralized. Perchance Windows work depends on this base.

## Validation
Fresh setup, build/start, API/UI access, model discovery, generation, conversion, plugin loading, shutdown, and update/restart on a supported Windows version.

## Complete when
A clean Windows system can install/build and complete a native Easy Diffusion generation with no Linux compatibility layer.
