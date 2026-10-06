# One shared GGML implementation

The authoritative source is the customized GGML already bundled with
`source/sdkit3-port-source/stable-diffusion.cpp`. The preparation script copies
that complete tracked tree and applies the reviewed patch in this directory.
Both stable-diffusion.cpp and llama.cpp consume that single CMake `ggml` target.
The original application snapshots remain the source of provenance.

Run from the repository root:

```sh
python3 cosmopolitan/shared-ggml/prepare.py --output cosmopolitan/out/source/shared-ggml
```

This requires Python 3, Git, and `patch` on the build host. `PIN.json` records the
complete customized source fingerprint, the exact llama donor files, and the
patch hash. Preparation refuses unexpected source changes. An unchanged tree
keeps its file timestamps. Output publication uses an exclusive lock and a
staging directory; `COSMOPOLITAN_SHARED_GGML.json` records the prepared contents.

## Preserved SD features

The SD tree's F8 E4M3/E5M2 types and conversions, convolution rotation and int8
operations, group normalization backward operation, Sage attention controls,
custom gradient implementations, and optimizer code remain in place. In
particular, the optimizer retains its graph-size override, per-parameter
learning-rate callback and userdata, and failure handling. The scheduler keeps
`ggml_backend_sched_set_allow_cpu_fallback`; the SD1.5 trainer's explicit refusal
of unsupported CPU fallback is not relaxed.

Every producer and consumer must compile with `GGML_MAX_NAME=160`. The parent
CMake project sets this globally and publishes it on `ggml` and `ggml-base`.
Add this prepared tree before the two applications and keep their
`*_USE_SYSTEM_GGML` options off so they reuse the existing target.

## Backports needed by the bundled llama revision

The donor is the existing llama snapshot at
`c589f0ed10c643678c4707dd160c21ac7633ebc0`. Only these interfaces and CPU kernels
are taken from it:

* `ggml_backend_dev_caps.mmap_support`, enabled for the CPU backend. Other
  existing backends conservatively leave the new capability false.
* `ggml_clamp_inplace`, and the donor's out-of-place `ggml_clamp` semantics.
  The latter now allocates a result instead of aliasing its input, which is
  required when llama retains both tensors. CPU clamp already supports this.
* `ggml_rope_set_offset`, its sixteenth operation parameter, and the CPU
  forward/backward RoPE kernel that preserves channels outside the rotated
  range. An offset of zero retains the existing layout. A separate local fix
  carries this offset into the autodiff builder's backward node; the donor's
  builder otherwise loses it. The same patch preserves zero-dimension RoPE as
  an identity operation: its channel-copy loop must not repeatedly skip an
  empty rotated range.
* `ggml_ssm_scan(..., ids, K)` and its CPU kernel. `K=1` retains the old packed
  output; larger K stores recent state history used by the llama Mamba2 path.
  The existing GGML operation test and Vulkan debug clone pass K explicitly.

The central backend support query rejects nonzero RoPE offsets and SSM K>1
for other bundled backends until their kernels are ported. This prevents the
scheduler from sending new tensor layouts to old GPU kernels. It does not turn
the current CPU integration into a validated GPU implementation. The existing
GPU limitations of the custom SD training graph also remain.

## Static backend registry

Compile the `ggml` target with `GGML_COSMO_STATIC_ONLY`. With that definition,
`ggml_backend_load_all` and `ggml_backend_load_all_from_path` initialize only the
linked backend registry. They do not scan the executable directory, the current
directory, or `GGML_BACKEND_PATH`. An explicit `ggml_backend_load` call returns
null and reports that dynamic backends are disabled. Merely setting
`GGML_BACKEND_DL=OFF` upstream does not disable those runtime searches.

The ordinary upstream loader remains available when this dedicated definition
is absent. Static registration can later include additional backends compiled
against this same GGML tree; arbitrary native GGML plugins cannot safely share
this application's customized types, operations, and tensor ABI.

## Focused compatibility checks

`compat_test.c` exports `cosmo_shared_ggml_selftest()` for the main application's
selftest. It checks CPU mmap capability, clamp output independence and explicit
in-place operation, offset RoPE against a scalar rotation reference, inverse
rotation and the actual autodiff graph, and Mamba2 K=1/K=3 packed output against
a double-precision recurrence. The SSM case includes grouped heads and
reordered state-pool IDs. Each case runs with one and four CPU threads.

The gradient check reproduced the donor offset-loss bug before the local fix.
These checks complement the application's existing custom SD backward and
optimizer tests; they do not validate an entire training run or GPU backend.

This merge establishes source and ABI agreement. Actual model inference,
partial SD1.5 training behavior, and each enabled backend still require their
own numerical and application tests; the presence of an API is not proof of
operator coverage.

## Persisted GGUF type IDs

The bundled llama donor and the shared SD GGML agree on the numeric IDs of all
35 tensor formats they have in common. No common format has a different ID,
and there are no donor-only formats. The shared tree preserves the two SD
extensions: `GGML_TYPE_F8_E4M3=43` and `GGML_TYPE_F8_E5M2=44`.

The only common enum name with a different value is `GGML_TYPE_COUNT`: 43 in
the donor and 45 in the shared tree. This is a bounds-check sentinel, not a
stored tensor format, so neither F8 extension collides with a donor format.
The retired gaps at 4/5, 31-33, and 36-38 remain unassigned in both trees. The
donor rejects files using the F8 IDs as out of range rather than interpreting
them as a different quantization.

All 13 GGUF metadata value types and `GGUF_VERSION=3` also agree. This audit
compared the actual enum values in both pinned headers; `gguf.cpp` reads and
writes tensor types as 32-bit integers using those values. It establishes
numeric format compatibility for these snapshots, separately from numerical
operator support and model-level validation.
