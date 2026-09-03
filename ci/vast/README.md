# Vast.ai runtime testing policy

Vast.ai is a paid runtime-validation target. GitHub Actions remains the build system.

## Non-negotiable rules

1. **Compile on GitHub, never on Vast.ai.**
2. `Test, Vast AI` must be selected explicitly from a manual API workflow dispatch.
3. The requester must type the exact approval value `APPROVE` and provide the Vast.ai offer ID before any paid API call is allowed.
4. Ordinary `compile`, regression, push, pull-request, scheduled, or failed-build paths must never create a Vast.ai instance.
5. The GitHub-compiled artifact is packaged on GitHub into a test image derived from the exact GHCR toolchain image used for that API build.
6. Vast.ai runs that packaged image. It must not compile or link the project there.
7. The rented instance must be destroyed in an `always()` cleanup step after success or failure.
8. `VASTAI_KEY` is a GitHub repository secret. It must not be written to the repository, artifacts, images, logs, or test output.
9. If an API target does not yet have a real containerized GitHub build image, its Vast.ai path must stop before the API call. Do not silently substitute another image.
10. Compile success and container smoke success are not equivalent to backend runtime validation. Backend/API-specific runtime tests should replace or extend the smoke command as executable tests become available.

## Current flow

```text
manual workflow dispatch
        |
        +-- mode = compile ----------------------> GitHub build only
        |
        +-- mode = Test, Vast AI
                |
                +-- GitHub compile
                +-- upload compiled artifact
                +-- require APPROVE + offer ID
                +-- package artifact into exact build/toolchain image on GitHub
                +-- push immutable run-specific GHCR test image
                +-- create Vast.ai instance with that image
                +-- run non-compiling runtime smoke/test command
                +-- destroy Vast.ai instance even on failure
```

The shared implementation is `.github/workflows/995-vast-ai-runtime-test.yml`.
