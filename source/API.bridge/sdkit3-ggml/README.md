# sdkit3 GGML API bridge

This Theory-only compatibility project owns the application adapters that map
stable-diffusion.cpp GGML tensors to API.cpp Common attention requests. It is
kept outside `source/sdkit3-port-source` so refreshing Easy Diffusion's moving
sdkit3/Leejet hybrid does not overwrite API.cpp integration code.

The embedded GGML tree still owns tensor definitions, scheduling, and native
fallback kernels. Its CUDA CMake file compiles the adapter sources here and
provides the narrow GGML include boundary. API.cpp Common and backend
translation layers do not include this bridge.

Current bridge-owned adapters:

- `ggml-cuda/xformers-attention.cu`: GGML to normalized xFormers Common.
- `ggml-cuda/flash-attention.cu`: GGML to normalized FlashAttention Common.

SageAttention remains in API.cpp's CUDA translation tree while its Common
migration proceeds. Once Sage has a complete Common `forward` route, its GGML
request adapter belongs here as well.

The bridge is source-compatible with the GGML snapshot recorded by
`source/sdkit3-port-source/SOURCE_SNAPSHOT.md`. A future GGML refresh must
compile the bridge before updating that record.
