# API.cpp

`source/API.cpp` normalizes backend-specific compute behavior behind one Library interface for Easy Diffusion.

The ownership flow is:

```text
backend definition -> backend translation -> feature common -> Library -> Easy Diffusion
```

- `common/` owns library-wide backend identifiers, capability checks, and registry/dispatch contracts.
- `features/` owns per-feature definitions, translations, common behavior, and prototypes.
- `library/include/api/` and `library/src/` own the public compiled Library surface.
- `cmake/` and `Makefile` remain at the API.cpp root.

For an exact directory contract and migration rules, see [LAYOUT.md](LAYOUT.md). Production CUDA attention is owned by the `cuda/translation/gpu` layer within the corresponding `flash`, `sage`, or `xformers` feature; legacy GGML CUDA paths are compatibility shims.
