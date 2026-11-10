# Portable compiler trial

Theory-only, standalone build setup for trying Cosmopolitan without changing
the application's existing compiler, installer, backend selection, or workers.
This is not the portable launcher implementation.

The first target reuses `../sdkit3-port-source/tools/native_image_tools.cpp`
without modifying or copying its source. It uses C++ and bundled stb headers,
but does not link GGML, LibTorch, Python, or a GPU SDK. Its normal SDKIT3 target
continues to exist unchanged.

From the repository root, with an extracted Cosmopolitan toolchain:

```sh
cmake -S source/portable-tools -B source/artifacts/cosmo-portable-tools \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchains/cosmopolitan.cmake" \
  -DEDCPP_COSMOCC_ROOT=/absolute/path/to/cosmocc-4.0.2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build source/artifacts/cosmo-portable-tools --parallel 2
```

The intended portable output is `bin/sdkit-image-tools.com`. Its companions
`sdkit-image-tools.com.dbg` and `sdkit-image-tools.aarch64.elf` are
architecture-specific debug artifacts, not substitutes for the APE file.
Do not strip the APE with host `strip`.

For a host-compiler comparison, configure the same source in a separate build
directory without the toolchain options. Never switch compilers inside an
existing CMake cache. Do not deploy either trial over the current backend.

No application runtime tests are registered or run by this project. See the
[migration assessment](../../docs/cosmopolitan-compiler-migration.md) for
validation status, compiler constraints, and the next launcher stage.
