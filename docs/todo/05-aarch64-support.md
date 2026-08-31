# 05 — aarch64 Support

## Objective
Make EasyDifussion-Cpp build and run correctly on 64-bit ARM Linux outside of Jetson-specific assumptions.

## Implementation
1. Audit x86-only compiler flags, AVX intrinsics, assembly, prebuilt downloads, binary names, and installer architecture checks.
2. Use GGML ARM/NEON implementations where available; conditionally compile x86 SIMD only on x86.
3. Add CMake/toolchain detection for `aarch64` and avoid assuming `x86_64` paths in scripts.
4. Build all native dependencies for ARM64 or obtain architecture-correct packages.
5. Audit alignment, atomics, mmap, endian assumptions, filesystem behavior, and subprocess handling.
6. Add an ARM64 build preset and CI/cross-compile job; still perform at least one native ARM test because cross compilation does not validate runtime behavior.
7. Keep JetPack-specific CUDA handling in the JetPack documents rather than polluting generic ARM support.

## Likely code areas
Top-level installers/scripts, `source/sdkit3-port-source`, stable-diffusion.cpp/ggml build flags, plugin/native helper build scripts.

## Fallback
CPU-only ARM64 must be a valid configuration even if no GPU backend is available.

## Validation
Clean configure/build, startup, API/UI, CPU generation with a small model, conversion utility, plugin loading, and shutdown on a generic ARM64 Linux environment.

## Complete when
A supported ARM64 Linux host can build and complete at least one generation path without x86 emulation.
