# Explicit source-runtime build consumer

The application build has an optional consumer for a **complete, prebuilt,
application-patched** source-runtime export. It does not complete the
foundation producer, upgrade a pin, build dependencies, or merge application
branches. No compatible export has been built or consumed in this checkpoint.

The default build remains on the existing stock SDK and foundation pin.
The new path requires all four options together:

```sh
python3 cosmopolitan/build.py \
  --foundation-handoff /export/HANDOFF.json \
  --foundation-root /export \
  --runtime-sdk /export/sdk \
  --software-webgpu-metadata /export/dependencies/LINK.json \
  --compiler-launcher none
```

This is an interface description, not a request to run a build or test.
The handoff must match the repository and **exact revision in the committed
`software-webgpu/PIN.json`**. There is no command-line pin override. The
current pin has not been changed to a source-runtime candidate; incomplete or
incompatible artifacts fail before compiler/tool setup. Ambient source-runtime
environment variables cannot silently activate the stock path.

`--prepare-only` stages application sources after verifying the full supplied
export. `--dependencies-only` verifies the supplied prebuilt dependency set and
does not rebuild it. `--libraries-only` compiles the linked application
libraries. The ordinary application build compiles and packages the artifact,
but this source-input path never launches producer probes, application
self-tests, inference or GPU tests. The source SDK and dependency producers
have their own compile-only controls and completion requirements.

## Exact inputs

First, the [handoff format](HANDOFF.md) binds the selected source identity,
compiler/runtime/header/CRT/C++ ABI inputs and all exported artifacts.
The consumer additionally checks:

- The selected SDK contains the exact bound `SOURCE_RUNTIME.json` and its
  canonical upstream runtime identity. Source repository/revision, ABI inputs,
  clean-source state, bootstrap role and fresh package-build declaration must
  match the supported producer contract.
- The SDK's `SOURCE_INPUTS.json` has its original raw-file hash bound by
  `SOURCE_RUNTIME.json`. Its complete tracked-source inventory hashes to both
  the producer's and handoff's `source_tree_sha256`; source tuples, ABI input
  fingerprints and package-recipe fingerprint must agree. The supported
  producer builds committed source directly, so `source_patches` is `[]` and
  `runtime.provenance.patches_sha256` hashes that canonical empty list.
- Every SDK file matches the original producer's size, hash and mode. Internal
  file and directory aliases are resolved from the producer inventory and
  must be fully materialized as ordinary exported files. Missing, extra,
  cyclic, dangling, escaping or special-file inputs fail.
- The selected x86-64 C/C++ drivers, archiver, ranlib, strip, APE linker and
  loader have their expected paths in that same SDK. The default x86-64
  `crt.o`, `libcosmo.a` and `libcxx.a` are explicit handoff runtime inputs.
- The SDK's source headers are the selected handoff header tree, not headers
  from the stock compiler bootstrap. The stock preprocessing-script patch is
  not applied to source-built drivers.
- The dependency `LINK.json` itself is a hashed handoff data artifact. It has
  the committed pin, selected SDK path, `runtime` equal to the upstream source
  runtime selection, static linkage, the x86-64 APE target and no shared-runtime
  library list. A stock `sdk_archive_sha256` must not label this source runtime.
- Mesa, LLVM and all four component provenance records declare that same
  source runtime. The dependency record contains the exact current
  `application_patches` map. This includes the bounded adapter-enumeration
  extension required by Easy Diffusion's WebGPU adapter; a generic foundation
  WGPU archive is insufficient.
- All runtime objects and delivered Mesa/LLVM/Rust archives are accounted for
  in the final link inputs. WebGPU currently delivers one selected staticlib.
  Includes, runtime source files, license files and other listed dependency
  bytes are verified, rather than accepting arbitrary search paths.

Paths passed through the current CMake JSON/list bridge cannot contain
semicolons, dollar signs, square brackets, or line breaks. Link switches are
limited to the reviewed archive-group/whole-archive switches, the source SDK's
existing C++/POSIX/math library flags, and simple symbol-alias definitions.
New switches require deliberate consumer support, not implicit host-library
search or a dynamic-library fallback.

## Relocatable dependency manifests

Producer `LINK.json` accepts either `path_mode: "absolute"` (also the legacy
default when the field is absent), or explicit
`path_mode: "export-root-relative"`. Relative mode uses normalized paths
within the verified export for exactly these selected fields:

- `sdk_root`, `wgpu_archive`, and `licenses`.
- Each `include_dirs`, `runtime_sources`, and `runtime_objects` entry.
- Non-option `driver_link_args` entries and every `file_sha256` key.

`license_sha256` keys remain relative to `licenses`. Nested original producer
records, their original paths and the source/runtime tuples are preserved;
they are evidence, not additional build search paths. Absolute entries,
parent traversal and unknown modes are rejected rather than guessed.

After all validation succeeds, the consumer produces an absolute CMake view
outside the immutable export. The view adds reserved `consumer_view` metadata
with `schema_version: 1`, `source_manifest_sha256`, `source_path_mode`, and
`export_root`. Original producer inputs must not contain `consumer_view`.
The generated bytes are deterministic sorted, indented, ASCII-escaped JSON
with a final newline. An existing modified view is rejected. Source bytes
are never rewritten, and generated views cannot be submitted as producer
evidence. This also applies to originally absolute manifests.

Moving a relative export preserves its original handoff, source LINK and
runtime identities, but changes the generated view and build selection
identity. That prevents a relocated export from reusing a CMake cache or
wrapper still pointing at its previous location.

## Rust standard-library delivery

Build input archives and final link archives are different things. The
dependency manifest must state `rust_std_linkage` explicitly:

- `embedded-in-wgpu`: both the `webgpu` and `rust_std` handoff components name
  the selected WGPU staticlib as their delivered archive. The Rust provenance
  record has `archive_delivery` equal to an object containing
  `mode: "embedded-in-wgpu"` and `container_sha256` matching that exact staticlib.
  Original Rust standard-library build inputs remain in the producer's
  provenance and may be separately hash-inventoried; they are not silently
  described as final link inputs.
- `separate-archives`: Rust component archives must all occur in the driver
  link list. Its provenance record has `archive_delivery` containing
  `mode: "separate-archives"` and `archives`, a mapping of the exact handoff
  relative archive paths to their SHA256 values.

These are bound producer receipts, not independent proof of Rust codegen,
archive-member equivalence after LTO, final ABI behavior or numerical
correctness. Runtime and model validation remain separate, unrun gates.

## Build isolation and provenance

The source path uses direct wrappers around its selected SDK programs and
chooses the APE loader from verified file format, not a filename guess. Its
compiler build directory and wrapper directory include a selection identity
covering the complete handoff bytes, original dependency manifest, generated
absolute view, export location and runtime identity.
Changing dependency archives or application patches therefore changes the
namespace even if the compiler/runtime itself is unchanged. Source and stock
objects do not share a CMake cache.
Ambient compiler/include/library overrides are not inherited by application
compilation. Host build utilities still come from the host environment.

Source-runtime sccache support is deliberately rejected until its own
preprocessing equivalence is qualified; the old stock-driver cache identity
cannot be reused. After compilation, SDK and dependency inputs are reverified
and their selection snapshot compared before packaging. The generated view's
bytes are rechecked too. Reports identify the source runtime, handoff, original
and generated dependency hashes, generated wrapper hashes and
static-only verification scope. They do not reuse historical validation
claims for new artifacts.

Exports must remain trusted and quiescent until compilation and packaging
finish. These checks do not lock against concurrent mutation or provide a
security boundary against malicious local writers. Content identities are
not signed attestations of the producer's claims.

## Verification status

Python syntax compilation, whitespace checks, static review and 91 non-model
handoff, consumer and argument-routing regression cases have passed. The
contract fixtures are synthetic, and argument routing mocks application work;
they do not execute the SDK/compiler, load a model or use GPU resources.
No real SDK has been accepted, no application/dependency build has run, and
no CPU/GPU model behavior is claimed verified. Existing UIs,
runtime feature implementations, workflows and dependency pins are unchanged.
