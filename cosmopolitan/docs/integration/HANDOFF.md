# Foundation handoff format 1

`cosmopolitan/foundation_handoff.py` is a host-side, standard-library-only
validator for a completed foundation export. It never launches a compiler,
application, runtime probe or model. The standalone validator does not select
a toolchain or update the application pin. A separate, explicitly selected
[build consumer](CONSUMER.md) can use a complete source-runtime export after
additional checks. The default application build and its pin remain unchanged.

The producer and consumer agree on exact input identities before changing
the compiler, runtime and dependency integration together. A successful
validation means that the declared lineage is consistent and exported file
bytes match their hashes. It does **not** independently prove the producer's
claims about source provenance, binary architecture, ABI correctness,
completeness of a compiler installation, final link ordering, or runtime
behavior. Those need the subsequent implementation and validation gates.

The export must be trusted and quiescent throughout validation and use.
Prepare it in an isolated directory, stop writers, and keep it unchanged until
the consumer has finished. This validator does not lock the export or provide
a security boundary against a process concurrently replacing files or parent
directories. Reverify after relocation or changes; never treat an earlier
report as authorization to consume subsequently changed bytes.

## Consumer interface

After the implementation-phase validation gate is opened, the consumer can
inspect an exported handoff with:

```sh
python3 cosmopolitan/foundation_handoff.py /export/HANDOFF.json \
  --root /export \
  --foundation-repository https://github.com/OWNER/REPOSITORY.git \
  --foundation-revision FULL_COMMIT_SHA \
  --architecture x86_64
```

The expected source repository and revision are separate arguments, not
values implicitly trusted from the handoff itself. The output explicitly
reports `static-identity-only`, `toolchain_selected: false` and
`runtime_behavior_verified: false`. Nonconforming input returns nonzero.
The library interface is `validate_handoff(data, root, expected_repository=...,
expected_revision=..., expected_architecture=...)`.

The validator can distinguish an `aarch64` export, but that does not add an
ARM application build. The current Easy Diffusion port remains x86-64.

## Exact schema

All named objects reject unknown or missing fields. Lists of artifact paths
reject duplicate entries. All required lists below are nonempty unless
explicitly stated otherwise.

The top-level object contains:

- `format`: exactly `easy-diffusion-foundation-handoff`.
- `schema_version`: integer `1`, not a boolean.
- `foundation`: `repository` (HTTPS URL, without credentials), `revision`
  (full lowercase 40- or 64-digit Git revision), `source_tree_sha256`, and
  `clean` (exactly true). The producer defines and records its complete source
  inventory in its provenance; the consumer pins the repository and revision.
- `target`: `architecture` (`x86_64` or `aarch64`), `abi` (exactly
  `cosmopolitan-ape`), and the actual `c_triple` and `rust_triple` used by the
  producer. Both triples must match the selected architecture; they need not
  equal each other.
- `runtime`, `toolchain`, `runtime_identity_sha256`, `components` and
  `artifacts`, defined below.

`runtime` contains:

- `kind`: `source-built` or `stock-sdk`. They are not interchangeable.
- `provenance`: for `source-built`, exactly `repository`, `revision`, and
  `patches_sha256`; for `stock-sdk`, exactly `version` and `archive_sha256`.
- `provenance_record`: a data artifact containing the upstream runtime/SDK
  identity record. Preserve the source producer's `SOURCE_RUNTIME.json`
  unchanged here; its upstream identity is evidence, not a substitute for
  the consumer's separately computed handoff identity. Materializing exported
  symlinks does not authorize rewriting that original provenance.
- `headers`: exactly `path` (the exported include tree) and `tree_sha256`.
- `crt_objects`: references to object artifacts supplying startup/CRT.
- `archives`: references to complete static runtime archives.
- `cxx_abi_archives`: references to the complete static C++ library/ABI
  archive set required by the compiler and application.

`toolchain` contains artifact references for `c_compiler`, `cxx_compiler`,
`archiver`, `ranlib`, `strip`, `ape_linker`, and `ape_loader`, all declared as
executable artifacts. `support_files` inventories the additional compiler
executables, support archives and data needed by those drivers. A wrapper
script alone is not a complete compiler export. The producer must include
the complete dependency closure, rather than relying on arbitrary host tools.

`artifacts` maps normalized, relative POSIX paths to objects containing
exactly `sha256` and `kind`. Kinds are `archive`, `object`, `executable`, or
`data`. Every artifact is read and hashed. Archive artifacts must have the
ordinary `ar` signature, complete member headers/payload lengths, valid padding,
and at least one non-index member, not an empty/thin archive or shared library.
This is a container check, not object-code validation; `.rlib` files may use
this kind. Executable artifacts must retain owner
execute permission. Paths cannot escape the explicitly supplied export root.
Materialize symbolic links when exporting: symlinks and non-regular files
are rejected, including within the header tree.

`components` contains exactly `rust_std`, `webgpu`, `mesa`, and `llvm`.
Each component contains:

- `version`: nonempty version or source identifier.
- `target_triple`: the exact Rust triple for `rust_std` and `webgpu`, or the
  exact C triple for `mesa` and `llvm`.
- `runtime_identity_sha256`: the same computed identity as the top level.
- `archives`: complete static artifact references.
- `support_objects`: additional object artifact references; may be empty.
- `provenance`: a data artifact containing the producer's build record,
  including its pins, build flags, patches and relevant link metadata.

The producer's provenance files are hash-checked, not interpreted as arbitrary
commands. Preserve their detailed identities and warnings. A top-level
component assertion does not retroactively prove how an archive was built.

These four components describe the existing embedded software foundation.
They do not represent all accelerator backend goals. Direct Vulkan remains
an important separate backend alongside WebGPU, including for older devices.
WebGPU's current ShaderF16 and subgroup constraints must not be generalized
to Vulkan. Native driver loading and direct GGML Vulkan integration retain
their own implementation, capability and validation requirements.

## Deterministic identity calculation

Canonical JSON means `json.dumps(value, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)` encoded as ASCII. The digest is SHA256 of those bytes.

For headers, hash canonical JSON mapping every regular file's relative POSIX
path within the header directory to its SHA256. Empty trees are invalid.
`header_tree_sha256(root, relative_path)` implements this convention.

For the runtime identity, hash canonical JSON containing exactly:

1. `foundation`, `target`, `runtime` and `toolchain` from the handoff.
2. `artifacts`: the union of all artifact records referenced by runtime
   `provenance_record`, `crt_objects`, `archives`, `cxx_abi_archives`, the seven
   toolchain executable fields, and toolchain `support_files`.

`runtime_identity_sha256(data)` implements this convention. Calculate it
after recording real artifacts and the header tree, then assign its result
to the top level and each component. The validator recalculates it before
checking component identity and all artifact bytes. Root paths do not enter
the digest, so an unchanged export can be relocated.

Unknown versions, dirty or unpinned sources, missing fields/archives/headers,
sentinel hashes, mismatched declared runtime/target identities, altered
artifacts and unsafe paths are errors. No partial acceptance or CPU-only
fallback is produced.

## Activation and test status

The current stock-SDK preparation path does not emit this new contract and
has not been silently relabeled as a source-runtime foundation. The optional
consumer requires a complete export matching the committed pin and selects
its compiler, headers, CRT, runtime archives, Rust delivery and final linker
together. No compatible source-runtime export has been built or consumed yet.

Contract tests are authored in `cosmopolitan/tests/test_foundation_handoff.py`.
They have not been run in this implementation checkpoint. No CPU behavioral
validation, WebGPU execution, GPU workflow or paid resource operation is
authorized by this document or by successful static identity validation.
