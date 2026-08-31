# Common API

Backend-neutral API contracts, capability definitions, shared types, and dispatch live here.

This layer must not contain backend-specific kernels. Backend implementations register or expose capabilities through the common contract, while callers use the common interface rather than directly selecting CUDA, ROCm, oneAPI, Vulkan, or CPU code.
