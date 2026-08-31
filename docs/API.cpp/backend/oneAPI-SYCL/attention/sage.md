# SageAttention — oneAPI / SYCL

## Status

**Not implemented in the common SageAttention API.**

The current SageAttention implementation is CUDA SM80-specific, while the HIP branch is an unsupported hook. No SYCL Sage kernel or SYCL adapter is wired into the backend-neutral Sage interface.

## Work required

A native SYCL/XMX or otherwise appropriate implementation needs its own capability checks and dispatch behind the common API. Do not route SYCL through CUDA-specific SM symbols.