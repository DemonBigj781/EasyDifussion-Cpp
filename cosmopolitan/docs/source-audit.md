# Earlier source updates: provenance and integration audit

Audit baseline: EasyDifussion-Cpp `main` at
`b42704a624652a53099fbc994e9a6b7648cd3b0c`, 2026-10-06.

The user reported that earlier attempts to update Easy Diffusion C++ and
llama.cpp might be unfinished. This audit separates a copied source snapshot
from an integrated, tested application. Snapshot metadata alone is not build
or runtime evidence.

## Method and exact results

Compared complete upstream Git trees at the recorded revisions against the
corresponding committed subtrees. Equality uses the Git blob object IDs, with
file modes compared separately. No truncated tree response was used. Full
paths and object IDs are in [source-audit.json](source-audit.json).

| Source tree | Upstream entries | Local entries | Identical content | Modified content | Added | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp at `c589f0ed10c643678c4707dd160c21ac7633ebc0` | 3,505 | 3,502 | 3,498 | 3 | 1 | 4 |
| stable-diffusion.cpp at `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`, excluding its GGML subtree | 390 | 423 | 351 | 36 | 36 | 3 |
| Diffusion GGML at `e20c3a14aa70ee84ca58499814206dd08d8026bc` | 2,157 | 2,209 | 2,130 | 25 | 54 | 2 |

Counts include non-source assets and Git submodule entries. They are not
percentages of implemented features. In addition to content differences,
llama.cpp has 121 mode-only changes and diffusion GGML has 18; these primarily
remove executable bits and do not change the files' bytes.

## llama.cpp update

The repository history records the snapshot import in
[`a6250a69015a09802a11281fb3ea3017f028858e`](https://github.com/DemonBigj781/EasyDifussion-Cpp/commit/a6250a69015a09802a11281fb3ea3017f028858e)
on 2026-08-30. The snapshot describes the 2026-08-29 upstream revision.

Only these files differ in content from that upstream revision:

- `cmake/build-info.cmake`
- `cmake/git-vars.cmake`
- `ggml/CMakeLists.txt`

They provide build identity for the de-gitted snapshot. The compiled llama
and embedded GGML source/header contents match the pinned upstream tree.
The four omitted files are a development hook configuration, a benchmark log,
an Apple framework build script and an Xcode workspace metadata file. None
is an omitted model engine source file.

`Theory` and `main` contain the same llama subtree object,
`a5c1a27ca8a7213a610fc83f452bc614dd07359a`. There is no newer llama source update
to recover merely by copying that directory from `Theory`.

**Finding:** no evidence of a partially copied llama core at the claimed pin.
This does not establish that it is the latest upstream version, that every
supported model works, or that the application integration is complete.

## Diffusion and training update

The recorded refresh is
[`d51cbf1b9e35321970d5889aff2851c6259c0b73`](https://github.com/DemonBigj781/EasyDifussion-Cpp/commit/d51cbf1b9e35321970d5889aff2851c6259c0b73),
"Update stable-diffusion.cpp while preserving native extensions", on
2026-09-05. Later commits through the audit baseline modify native inference,
model loading, UI integration, tools and training.

This tree is intentionally not an untouched upstream import. Modified and
added code covers model formats/loaders, conditioning, LoRA, UNet/CLIP, VAE,
attention, caching and the custom SD 1.5 trainer. The GGML changes include
custom math and optimizer behavior. These differences must be preserved and
tested, not discarded as drift during a refresh.

The three absent stable-diffusion.cpp entries are upstream submodules:
`examples/server/frontend`, `thirdparty/libwebm` and `thirdparty/libwebp`.
The local CMake configuration detects optional codec availability. Consequently
the source refresh is not evidence of WebP/WebM export coverage in the native
Cosmopolitan package. The Easy Diffusion C++ UI has a separate source tree.
The two missing GGML entries belong to the MNIST web example; no `src/` or
`include/` entry is missing in any of the three audited trees.

## Confirmed unfinished integration

The existing native CMake explicitly builds llama-server separately because
llama.cpp and diffusion pin incompatible GGML implementations. That protects
the ordinary multi-process build, but does not meet this branch's shared
runtime requirement. The Cosmopolitan build must reconcile the actual APIs
and custom behavior, then compile a single authoritative GGML target.

The ordinary application's conversion and orchestration paths also retain
Python. Native training is documented as partial. LibTorch/ONNX tools,
application API parity, queues, cancellation and UI behavior need their own
coverage; copying or compiling the model libraries does not complete them.

Existing workflow names are not proof of completion. For example,
[`009-cpu-complete-compile.yml`](../../.github/workflows/009-cpu-complete-compile.yml)
currently prints that its compile stages are reserved, and the CPU API
workflow builds its active implementation on `Theory`. The history query for
the llama import returned no Actions runs. The diffusion refresh query
returned one failed toolchain workflow, not a successful application runtime
test. These observations do not rule out unrecorded local tests; they do mean
the port needs fresh, explicit build and runtime evidence.

## Decisions for Cosmopolitan Test

1. Keep the current main baseline, including the user's recent custom work.
2. Stage portability changes separately from the preserved source snapshots.
3. Reconcile GGML from the custom diffusion tree, adding the required llama
   interfaces and semantics rather than linking both implementations.
4. Use one consistent tensor layout, including `GGML_MAX_NAME=160`.
5. Validate real llama inference, quantized tensor operations, existing
   training math, server startup and embedded UI using the same executable.
6. Track actual image generation, end-to-end training, application API parity
   and every accelerator independently of those initial checks.

An upstream refresh after this baseline must repeat the preservation and
compatibility audit. A successful compile alone cannot demonstrate that
custom training gradients, model routing or application behavior survived it.
