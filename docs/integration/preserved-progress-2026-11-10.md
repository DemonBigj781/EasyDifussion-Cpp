# Preserved-progress integration, 2026-11-10

## Scope and immutable inputs

This is a development integration, not a completed native-port or GPU validation.
The preservation branches remain unchanged and are temporary source records until
their useful work is integrated and verified.

| Input | Commit | Treatment |
|---|---|---|
| dev/cosmopolitan-foundation-integration | 24c8608 | Base foundation consumer, source inventory and link validation |
| dev/cosmo-gpu-component-inputs | 13d064e | Base plus split diffusion/text-encoder/VAE inputs |
| progress/cosmo-mint-2026-11-10 | 673d457 | Merged, including parity audits, mask preview and color-guide editor |
| dev/mint-main-preservation-20261110 | 6be1a9f | Merged, including both UI implementations, color-guidance backend, SAM and model-browser work |
| progress/theory-mint-2026-11-10 | 4755b7a | Preserved reference; not wholesale merged |

## Integration order and decisions

1. Foundation first, then its single GPU-component-input commit.
2. Merge the Cosmo snapshot. It shares the recent Cosmo baseline and does not
   conflict with the foundation changes.
3. Merge Mint main before publishing the increment: Cosmo's color-guide UI alone
   has no corresponding backend in the foundation base. Mint supplies the
   independent color-conditioning path and tests. Keep ordinary mask inpainting
   and color guidance separate, and retain the legacy and C++ UIs.
4. Three UI files conflict. Retain Mint's newer model browser HTML/JS, including
   automatic model-bound galleries, independent paging, tag controls and direct
   public-image browsing. Retain Cosmo's compatibility global-images route as
   model-location search; do not reintroduce the removed images_v6 search.
   Combine both sides of native-image-tools: SAM segmentation, mask preview,
   color-guidance preview refresh and masked-object removal all remain.
5. Review Theory by subsystem, not a wholesale merge. Relative to the foundation,
   its history is 379/711 divergent commits; 476 paths under DiffUser.cpp,
   INFERENCE.cpp, RestAPI.cpp and API.test are Theory-only in the combined tree.
   Next increment: inventory typed load/unload and Common routing contracts,
   adapt API.test fixtures to current routes, then evaluate backend dispatch
   implementations with compile tests. Do not replace current runtime ownership
   or declare backend support from directory names. Existing parity audit and
   docs/todo/33-cosmo-main-theory-parity.md provide the inventory framework.

Mint main was 36/4 divergent commits against foundation; the Cosmo snapshot
was 5/1. File counts alone are not behavior parity evidence.

## Validation in the cloud workspace

- Foundation suite: 117 tests passed, no skips, using existing host CMake/Ninja,
  compiler and archiver. Synthetic compile/link fixtures only.
- Audit, shared GGML, online-model-browser and UI integration: 46 tests passed
  using system Python with requests and scripts on PYTHONPATH.
- Native image tools, color-weighted inpainting, SAM catalog and segmentation
  contracts: 18 tests passed using the primary Python runtime.
- Sampling optimization contracts: 4 tests passed. Repaired an isolated AST test
  fixture to supply its unrelated interposer helpers; production behavior was
  not changed by that fixture repair. Requests were mocked, no inference ran.
- JavaScript syntax checks passed for the three resolved/affected plugin files.
- git diff --check passed.
- Civitai browser suite attempted, not passed: bundled Playwright browser absent;
  installed Chromium also failed at startup with socket Operation not permitted.
  The suite now accepts CHROMIUM_PATH for a supported host installation.
- Full application build, both live-UI flows, model inference, GPU execution,
  remote CI and packaged Windows/Linux runs are not validated by this increment.
  No RTX 3090, video LoRA testing, or Media drive inspection was performed.

## Remaining gates

Run both live UI/inpainting suites and the Civitai fixture browser suite on a
permitted browser host. Build the complete native application and execute
explicitly authorized CPU/model tests before claiming runtime readiness. Track
Theory ports separately with behavior contracts and compile evidence. Keep all
source branches until the final integration has passed these gates.
