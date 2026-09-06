# Common xFormers contract

`xformers.hpp` is the backend-neutral xFormers interface. It defines tensor metadata, masks, ALiBi, attention sinks, capabilities, normalized validation results, the staged score buffer, and the translation callback table.

Backend translations self-register by `Backend`. A caller queries capabilities or submits an `AttentionRequest` to Common. Common locates the registered translation, performs normalized validation, and dispatches either complete `forward` or an explicitly supported staged callback.

CPU currently exposes staged operations and complete forward orchestration. CUDA exposes only fused forward: `qkt()` intentionally returns false for CUDA so Common never forces a fused GPU implementation to materialize its score matrix.

Common must remain free of CUDA, HIP, SYCL, vendor handles, device launches, and backend memory ownership. Those details belong below the translation boundary.
