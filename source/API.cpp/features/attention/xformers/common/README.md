# Common xFormers code

This directory is the unified xFormers layer shared across backends.

Unlike backend directories, `common/` does not contain a `definition/` subdirectory. Backend-specific translation happens first under `[backend]/definition/[method].cpp`; once those definitions are understood, the unified implementation for each method lives directly here as `common/[method].cpp`.

Current phase: structure only. Do not implement unified behavior until backend definitions are filled and compared.
