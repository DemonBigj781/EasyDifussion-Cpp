# Cosmopolitan compiler migration and portable launcher

## Decision — 2026-09-20

Begin with an opt-in, isolated Cosmopolitan build. The intended end result is
an APE launcher that selects a compatible set of native executables and retained
Python workers. This does not replace every backend compiler, eliminate Python,
or make the Theory branch a runnable release.

Windows x86-64 is the initial Windows target. Dedicated Windows ARM development
and validation are deferred until viable, not prohibited. Do not add an ARM
rejection just to enforce this scheduling decision, and do not advertise
unvalidated emulated execution as native Windows ARM support.

## What swapping compilers means

The frontends and archive tools must change together in a fresh build tree:

| Build role | Cosmopolitan choice |
| --- | --- |
| C compiler | `cosmocc` |
| C++ compiler/link driver | `cosmoc++` |
| Static archive creation | `cosmoar` |
| Static archive indexing in our CMake toolchain | `cosmoar s`, for both architecture archives |
| Multi-OS/multi-architecture executable packaging | `apelink`, normally invoked by the compiler wrapper |

The upstream toolchain also supplies architecture-specific drivers for projects
whose configuration or assembly cannot use the fat wrapper. Those require
separate architecture builds and an explicit final APE link. Merely changing
the file extension or wrapping an existing CUDA/Windows/Linux executable does
not recompile its dependencies or make it portable.

The [opt-in CMake toolchain](../cmake/toolchains/cosmopolitan.cmake) selects
`Generic` rather than pretending the fat target is the build host's OS. It uses
static compile/archive checks, disables RPATH, and limits target package/header/
library discovery to the Cosmopolitan prefix. The final target build still has
to link successfully. Portable code must use runtime OS detection; CMake's
`WIN32`/`APPLE` and the build host's `uname` cannot choose the deployed host.
Do not mix host-built objects, libraries, LTO artifacts, or cached feature
results into this build. Avoid `-march=native` and architecture-specific flags
that accidentally raise the launcher's minimum CPU requirements.

The supplied language-rules override is required: CMake's `Generic` platform
otherwise names objects `.obj`, which the fat compiler can misclassify during
linking. The trial exposed this as a misleading successful link without the
intended helper code. Using `.o` removed that warning; inspect actual payloads,
not just a successful build exit status.

## Current repository boundaries

| Component | First-step treatment | Evidence / reason |
| --- | --- | --- |
| Portable launcher | Standalone experimental target | [Autolauncher](../source/autolauncher/README.md) implements selection and supervision without GPU/Python linkage. |
| `sdkit-image-tools` | Isolated compiler-swap trial | C++ plus bundled stb headers; no GPU or LibTorch dependency. |
| SDKIT3 server | Keep native build | Its CMake enables shared stable-diffusion/GGML libraries, OS-specific RPATH, and bundling rules. |
| Bundled llama-server | Keep native, separate process | SDKIT3's `ExternalProject_Add` explicitly forwards its C/C++ compilers. A global swap would propagate here too. |
| CUDA / ROCm / SYCL / oneAPI CPU | Keep their selected native toolchains | `install.sh` chooses CUDA tooling, ROCm Clang, or Intel `icx`/`icpx`; these are not replaced by `cosmoc++`. |
| LibTorch vision worker | Keep native | Links the selected LibTorch distribution and possibly its CUDA runtime. |
| INFERENCE Common code | Later audit candidate | Its static C++ target alone does not establish Cosmopolitan compatibility; do not sweep unfinished code into this trial. |
| Python application/conversion/training workers | Retain Python | Follow the [uv and freezing policy](python-runtime-policy.md); a portable launcher does not replace their APIs. |

Read-only evidence: [SDKIT3 build](../source/sdkit3-port-source/CMakeLists.txt),
[installer](../install.sh), [native build wrapper](../source/Build.sh), and
[image helper source](../source/sdkit3-port-source/tools/native_image_tools.cpp).
This assessment does not promote API.cpp scaffolds into supported backends.

## First compiler trial

Use the standalone [portable-tools build](../source/portable-tools/README.md).
It recompiles an existing lightweight component without entering SDKIT3's
top-level build or changing any default application target. This separates a
compiler-compatibility check from launcher implementation.

Use a versioned upstream toolchain archive, starting with `cosmocc-4.0.2.zip`,
not an unpinned `cosmocc.zip`. Record the downloaded archive's SHA-256 with build
provenance; a locally calculated hash is not an independently verified upstream
signature. Keep toolchains and generated files outside tracked source. The
configuration does not download tools, install a loader, alter `binfmt_misc`,
or modify global `CC`/`CXX` settings.

Upstream's archive wrapper has limitations around spaces in object paths and
response-file inputs. Use simple build paths for this trial and explicitly
test these cases before broadening the build. Do not silently drop ARM companion
objects or substitute the host archiver to work around an error.

## Launcher implementation and release integration

The standalone [autolauncher](../source/autolauncher/README.md) now implements
schema-1 bundle selection, runtime OS/ABI/CPU checks, SHA-256 verification,
bounded accelerator probes, ordered process startup, TCP readiness, exit-code
propagation, and cleanup. The build-time packager emits manifests from staged
artifacts. An optional native CUDA-driver probe is built separately. The
following integration requirements still apply to actual release bundles;
existing application startup/worker ownership has not been replaced.

1. Build the launcher separately with Cosmopolitan; build accelerated workers
   using their native toolchains and separate build directories.
2. At final packaging, generate a versioned manifest from actual built files:
   component role, relative executable/library paths, compatible bundle/protocol
   version, OS/architecture/CPU requirements, accelerator/driver requirements,
   hashes, working directory, and required uv project/interpreter/lockfile.
   Existing SDKIT3 post-build copy hooks are an integration point, but packaging
   must refresh metadata even when an unchanged target did not relink.
3. Detect the deployed host at runtime. Use small native probes where GPU/driver
   checks need vendor APIs; OS/architecture detection alone is insufficient.
   Keep manifest selection consistent with the existing Python backend selector.
4. Select a compatible component set, honor explicit overrides, and explain
   missing dependencies. CPU fallback is only valid for operations that support
   it. The absence of a compatible artifact is not a Windows ARM policy ban.
5. Launch through an argv-based process API such as `posix_spawn`, not a shell
   command assembled from user data. Validate paths within the trusted bundle,
   preserve arguments/Unicode, and handle readiness, output, cancellation,
   process-tree shutdown, and exit status.

Keep Python workers external and uv-managed with pinned interpreters and locks.
PyInstaller packaging remains an option for a worker/controller; it is not
needed to provide the Cosmopolitan launcher itself. No inference, training,
model conversion, or driver installation belongs in the launcher.

## Validation gates and current status

- **Compile-only results, 2026-09-20:** on a Linux x86-64 host with CMake
  3.31.6, both the native GCC 14.2 build and Cosmopolitan 4.0.2 / GCC 14.1 build
  configured and linked the unchanged image-helper source successfully. The
  final portable build produced `sdkit-image-tools.com` (2,423,543 bytes) with
  the APE `MZqFpD='` magic. Its companion ELF payloads identify as x86-64 and
  AArch64; the expected image-helper diagnostic string was found in both.
- C/C++ compiler checks used `cosmoar` for archive creation and indexing.
  Repeat configuration and incremental build passed. Missing toolchain root,
  invalid root, and conflicting compiler configurations failed as intended.
  Both native and Cosmopolitan builds emitted an existing warning in vendored
  `stb_image.h:5166` (`-Wstringop-overflow`); that source was not changed.
- Toolchain archive SHA-256, observed from the versioned GitHub release:
  `85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44`.
  Trial APE SHA-256:
  `6ecf8749cc85316999701596e0671f53c465f6adbf7e1e585a8acd3a00d79e09`.
  These identify this trial, not a signed release or a runtime validation.
- The autolauncher, packager and optional native CUDA probe are implemented.
  Native Linux x86-64 and APE/Linux x86-64 fixture tests exercise selection,
  argument handling, environment/cwd, probe fallback/timeout, integrity checks,
  readiness and descendant cleanup. These are isolated launcher tests, not
  Theory application or GPU inference validation. See its README for commands
  and remaining platform/release-integration limitations.
- Record configuration, compile/link results, artifact format/architectures,
  and archive hash separately from runtime results.
- Theory remains a design/coding branch. Do not start the application, install
  trial artifacts, run inference, or infer platform support from compilation.
- Before release promotion, test native Linux x86-64/ARM64, macOS x86-64/ARM64,
  and Windows x86-64 execution, driver probing, Unicode/spaces, failure paths,
  clean shutdown, and missing/offline Python environments as applicable to the
  claimed support matrix. Windows ARM remains deferred, without a hard block.
- Test upstream's APE shell/loader, Linux executable-handler, and WSL caveats.
  Do not change global executable handlers or disable WSL interoperability as
  an automatic installation shortcut. Provide a documented launch fallback.

## Primary references

- [Cosmopolitan toolchain](https://github.com/jart/cosmopolitan/blob/master/tool/cosmocc/README.md)
- [Upstream compiler/CMake guidance](https://github.com/jart/cosmopolitan/discussions/1341)
- [Cosmopolitan archive wrapper](https://github.com/jart/cosmopolitan/blob/4.0.2/tool/cosmocc/bin/cosmoar)
- [Portable process-spawn example](https://github.com/jart/cosmopolitan/blob/master/examples/spawn.c)
- [APE architecture specification](https://github.com/jart/cosmopolitan/blob/master/ape/specification.md)
- [Platform caveats](https://github.com/jart/cosmopolitan#platform-notes)
- [CMake toolchains](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)
