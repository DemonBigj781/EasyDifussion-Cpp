# API.cpp

`source/API.cpp` normalizes backend-specific compute behavior behind one Common interface for Easy Diffusion.

The runtime call flow is:

```text
Easy Diffusion <-> Common exported API <-> backend translation <-> backend definition <-> native runtime/device
```

Results return through the reverse path.

- `common/` owns library-wide backend identifiers, capability checks, and registry/dispatch contracts.
- `features/` owns per-feature definitions, translations, common behavior, and prototypes.
- `library/include/api/` and `library/src/` package the exported Common
  surface. They own no backend behavior and may call only Common.
- `cmake/` and `Makefile` remain at the API.cpp root.

For the directory contract and migration rules, see [LAYOUT.md](LAYOUT.md). For
the audited distinction between active, compatibility, prototype, superseded,
and future-only paths, see [LAYOUT_AUDIT.md](LAYOUT_AUDIT.md). Production CUDA
attention is owned by the `cuda/translation/gpu` layer within the corresponding
feature. The Theory-only `source/API.bridge/sdkit3-ggml` xFormers and Flash
adapter files create Common requests; they do not call CUDA translations
directly. Flash retains an optimized GGML fallback for unselected or unsupported
Common requests. Sage selection now uses a normalized Common `support` request,
but its forward launch is still the GGML-native SM80 compatibility path and
remains routing debt.

Backend implementation and validation maturity is tracked in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
