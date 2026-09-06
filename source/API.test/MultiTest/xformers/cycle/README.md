# xFormers cycle tests

The CPU and CUDA cycle sources are active and are built by
`source/API.test/Makefile`. They call the xFormers Common API and also exercise
the normalized model-byte load/unload route.

The Mesa, OpenGL, and Vulkan sources, `xformers-load-unload-test.hpp`, this
directory's `CMakeLists.txt`, and the exact filename
`xformers-load-unload-test .cpp` are reserved and unwired. No xFormers or model
lifecycle support for those backends follows from these placeholders. Preserve
the spaced filename until its intended role or rename is explicitly decided.
