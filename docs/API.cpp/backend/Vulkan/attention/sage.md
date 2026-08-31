# SageAttention — Vulkan

## Status

**Not implemented.**

The current native SageAttention implementation is CUDA SM80-specific. No Vulkan compute-shader implementation is wired behind the backend-neutral Sage API.

A Vulkan implementation would require a backend-appropriate algorithm/shader path plus capability and datatype/shape checks; CUDA SM-specific code must remain isolated.