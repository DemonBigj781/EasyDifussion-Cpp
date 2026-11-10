# Definition ownership correction

Date: 2026-09-08
Base Theory commit: `9764a79c65115b6336e84ed106a9fe8d71993ed6`

## Scope

This working-tree correction addresses ownership findings C-01, C-02, and
C-03 from `FULL_COMMON_CONTRACT_REAUDIT.md`.

## Corrected routes

- CPU xFormers staged math and forward sequencing now live in
  `features/attention/xformers/cpu/definition/cpu/`. Matching translations
  convert Common requests and score buffers, invoke definitions, convert
  results, and register callbacks.
- CUDA xFormers pointer/device checks, CUDA runtime calls, fused kernel, launch,
  and synchronization now live in
  `features/attention/xformers/cuda/definition/gpu/forward.cu`.
  `cuda/translation/gpu/forward.cu` remains the authoritative Common adapter.
- CPU FlashAttention validation and execution now live in
  `features/attention/flash/cpu/definition/cpu/`; its translation is an
  adapter and registration owner.
- CUDA FlashAttention pointer/device checks, CUDA runtime calls, fused kernel,
  launch, and synchronization now live in
  `features/attention/flash/cuda/definition/gpu/forward.cu`; its translation
  contains only conversion, definition calls, result conversion, and
  registration.
- Flex Common now owns only the normalized request/result, validation,
  registry, dispatch, and compatibility wrappers. The CPU translation adapts
  the contract, and the CPU definition owns selection, margin removal,
  normalization, thresholding, pooling, and expansion.
- Overflow translations now prepare normalized capability information.
  Common constructs and walks the candidate plan before dispatching one
  candidate to a translation. CUDA translation no longer calls Common planning.

Explicit Make and GitHub Actions source lists were updated for the relocated
xFormers and Flash definition sources and the new Flex Common route test.

## Validation status

Validation was authorized after the source relocation. It used GCC 14.2,
CMake 3.31.6, CUDA toolkit 12.4, and an RTX 3060 (`sm_86`, driver 580.94.18).
All generated files were written below the fresh directory
`/tmp/easy-diffusion-theory-ownership-20260908.G9OBdz`; existing repository
build trees were not used or cleaned.

Passed:

- CPU xFormers Common semantic/lifecycle suite, including 32 load/forward/
  unload cycles and rejection cases;
- CPU FlashAttention self-contained Common route;
- CPU Overflow allocation/release and mixed-backend planning;
- CUDA xFormers Common semantic/lifecycle suite, including F32/F16/BF16,
  masks, causal mode, ALiBi, soft-cap, GQA/MQA, strides, sinks, and 32 cycles;
- CUDA FlashAttention numerical route and 32-cycle Common API roundtrip;
- regular CUDA Overflow and the opt-in stress route, including a real
  `cudaMalloc` OOM followed by a GPU-verified 2048 MiB managed fallback;
- the broader `make check`, `check-xformers`, `check-flash`, `check-flex`, and
  `check-sage` API regression targets;
- Compute Sanitizer memcheck for xFormers CUDA, Flash CUDA, and the Flash CUDA
  roundtrip, each with zero reported errors;
- the CPU Flex public Common route, covering capabilities, deterministic
  selection/expansion, compatibility wrappers, and rejection cases;
- DiffUser/SDKIT3/llama and INFERENCE dependency-boundary scripts, plus all
  four privacy-debug regression tests;
- `git diff --check` and the static ownership inspection described below.

The first build exposed two correction-local defects, both fixed before the
passing runs: two stale arguments at Overflow's diagnostic aggregator and
anonymous-namespace linkage on the moved CUDA native entrypoints.

Static inspection found no CUDA headers, CUDA runtime calls, kernel declarations,
or launch syntax in the corrected attention Common/translation paths, and no
Overflow planner call in a backend translation.

End-to-end application maturity was not re-established. The current
stable-diffusion.cpp/ggml snapshot successfully compiled to static libraries,
and the shared image harness was updated for upstream's new `generate_image`
out-parameter API. The generator then failed to link because commit `d883d6e`
had intentionally removed the old GGML-to-DiffUser adapters and their
`ggml_cuda_xformers_attn_launch_count`/
`ggml_cuda_flash_common_attn_launch_count` observation hooks. Those deleted
bridges were not restored implicitly. Current xFormers and Flash claims stop at
`R`, not `E`.

Findings C-04 (stronger automated boundary enforcement) and C-05 (native and
console log privacy) are separate follow-up work and are not claimed as fixed
by this ownership correction.
