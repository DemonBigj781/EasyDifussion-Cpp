# Compiler caching for the application build

The production application recipe supports sccache 0.16.0. The main
[Cosmopolitan workflow](../.github/workflows/cosmopolitan-test.yml) enables it
for actual application C/C++ compilation and for Rust dependencies when they
need compilation. The existing Cosmocc 4.0.2 compiler wrappers, Rust custom
target, patched standard library, runtime configuration and worker limits
remain authoritative.

This is separate from the earlier [toolchain compatibility experiment](vast/BUILD_CACHE.md).
That experiment demonstrated cache hits; it did not enable caching for the
application. The production integration records its own statistics and
elapsed times. Its first full CI build failed during compilation; the
[CI build record](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37629231689)
records 97 cache misses, zero hits and no completed application. It does not
establish a whole-application speedup.

## Build commands

Install sccache 0.16.0 and make it available in `PATH`, then run:

```sh
python3 cosmopolitan/build.py --compiler-launcher sccache --jobs 2
```

An absolute path to that executable is also accepted. The equivalent
configuration variable is `COSMO_COMPILER_LAUNCHER=sccache`; an explicit flag
takes precedence. The local default is `none`. To switch an existing CMake
application build back to uncached compilation, use
`--compiler-launcher none`; the recipe clears both cached CMake launcher
settings on reconfiguration. Independently supplied environment wrappers
remain the caller's responsibility when the recipe's cache option is off.

Enabling the option adds exactly one CMake C/C++ compiler launcher before the
existing `out/toolchain/cc` and `out/toolchain/cxx` wrappers. It does not change
`CC`, `CXX`, the target compiler, archive tools or linker. It sets
`RUSTC_WRAPPER` to the same sccache executable and `CARGO_INCREMENTAL=0` in the
environment passed to dependency preparation. Conflicting launchers and a
nested `RUSTC_WORKSPACE_WRAPPER` are rejected. The Cosmocc SDK archive hash is
included in `SCCACHE_C_CUSTOM_CACHE_BUSTER` because the small wrapper file alone
does not identify all the compiler bytes it invokes. The effective namespace
also includes `preprocess-parity-v1`; the recipe accepts the workflow's SDK
base value and adds that revision. This excludes entries written before the
preprocessing correction below.

Cosmocc 4.0.2's standalone preprocessing branch omits its normalization header
and compilation flags. An uncorrected sccache preprocessing call can therefore
see host Linux macros, an incorrect numeric Cosmopolitan version and different
integer typedefs. The application wrappers generate a preprocessing shim from
the pinned SDK driver with one checked change: `-E`, `-M` and `-MM` receive the
same normalization header, mode-dependent flags and frame-pointer option as
compilation. The original compiler path remains shell `$0`, preserving the
SDK's directory, architecture and language selection. The SDK itself is not
modified, and ordinary compilation continues through its original driver.
Reports record the original driver, generated shim and wrapper hashes.

The application contains no C++ module units, so its CMake configuration
disables module scanning. Otherwise CMake's GCC C++20 scan adds `-fmodules-ts`,
which sccache 0.16.0 treats as non-cacheable; this accounted for 183 uncached
calls in the failed first full build.

The main workflow installs the reviewed
`mozilla-actions/sccache-action@fc920bf0ec8de6ee65d409111f7ec508035751ba`
(v0.0.11), requests binary version `v0.16.0`, and enables its GitHub Actions
cache backend. It also exports the Rust wrapper before the staged dependency
commands. The pinned SDK and completed LLVM/Mesa/Rust dependency caches are
still restored and saved as before; their reuse avoids work independently of
compiler-cache hits.

## Observe the real build

Each recipe invocation writes `out/build-cache/<mode>.json`, where mode is
`prepare`, `dependencies`, or `application`. Reports contain the exact launcher
version and hash, full before/after sccache statistics, per-language hit/miss
deltas, success/failure status and measured phase timings. Build errors remain
errors; the report is also written when a compilation fails. The application
link metadata records the cache configuration.

CI additionally preserves `ci-before.json`, `ci-after.json` and
`ci-summary.json` in the build-diagnostics artifact and prints the measured
counters and timings in its job summary. Deltas use the same server without
resetting its statistics. The overlapping `adv_counts` breakdown is not added
to the language totals. Inspect the raw statistics for non-cacheable calls or
unsupported invocations; hit/miss totals alone are not total compiler requests.

A restored dependency closure can result in **no Rust compilations** during a
run. Zero Rust hits and misses then represent skipped work, not evidence that
the Rust cache failed or was exercised. These timings have no uncached full
build baseline, so they are not by themselves a speedup ratio.

## Optional replay of actual application objects

To measure real cache retrieval without repeating image inference or a full
clean application build:

```sh
python3 cosmopolitan/build.py --compiler-launcher sccache --cache-replay --jobs 2
```

After the normal application build, this diagnostic removes and recompiles
only the application's `math_compat.c` and `config.cpp` objects at their same
paths using their existing Ninja rules. Each must produce a positive cache-hit
delta and the original SHA-256 object hash. The original object is restored if
compilation or validation fails. No extra application link or model test is
performed by the replay. Its source names, object identities, individual
before/after counters and elapsed times are included in the application report.
The main CI build enables this narrow diagnostic once per build; local builds
run it only with the flag.

## Coverage boundaries

The explicit launcher covers C and C++ targets in the application CMake build.
The inherited Rust wrapper covers compilations invoked through the pinned
Rust runner. It does not replace that runner or its target/std/cfg settings.

When this cache option is enabled, independently supplied
`CMAKE_C_COMPILER_LAUNCHER`/`CMAKE_CXX_COMPILER_LAUNCHER` environment settings
are rejected with an instruction to use the application option. This avoids
silently enabling sccache for dependency compilers that do not use the corrected
application wrappers. The Rust wrapper remains inherited by dependencies.

There is no automatic coverage claim for Mesa's Meson compilation, direct
compiler calls, assembly, linking, packaging, runtime shader compilation or
model execution. LLVM has its own configuration fingerprint and cached CMake
build; this change does not force that configuration to adopt a launcher.
Changing a CMake-launcher environment variable alone is insufficient for an
already configured dependency tree. The application recipe passes its launcher
settings explicitly on every reconfiguration. Existing externally configured
dependency caches remain the caller's responsibility; the recipe does not
rewrite them or claim they have been validated with this launcher.
