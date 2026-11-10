# 33 — Cosmo Main/Theory Parity and Native One-Executable Completion

## Objective
Finish the `cosmopolitan-test` branch as one native Easy Diffusion executable while preserving the active main application contract and selectively porting useful Theory work. The target is not a Python launcher, subprocess bundle, or proof-of-concept tensor probe: supported Windows/Linux x86-64 users should run the same application bytes, use native UI/API features, and only need Python for build-time tooling or explicitly documented temporary fallback work.

## Current audit baseline

- Active main source: `/mnt/38FEF88DFEF84522/easy-diffusion-versions/easy-diffusion-cpp/easy-diffusion-cpp`, branch `feat/modern-optional-plugin-ports`, head `b42704a`, 7,924 tracked files.
- Theory source: `/mnt/38FEF88DFEF84522/easy-diffusion-versions/easy-diffusion-cpp/easy-diffusion-cpp-Theory`, branch `Theory`, head `8315a81`, 8,290 tracked files.
- Cosmo target: `/mnt/38FEF88DFEF84522/easy-diffusion-versions/easy-diffusion-cpp/Easy-Difussion-Cpp-Cosmo`, branch `cosmopolitan-test`, head `0f10cd4a`, 8,538 tracked files.
- Active main is nearly file-level present in Cosmo. The only tracked active-main path missing from Cosmo at this audit is `.github/workflows/##0-All-Toolchain-Image.yaml`.
- Theory has large material not present in Cosmo: 449 `source` paths, 152 `docs` paths, 7 `Audit` paths, 7 `.dockerfiles` paths, 2 `ui` paths, 2 `scripts` paths, 1 `ci` path, and 1 `.github` path.
- Cosmo still contains 124 tracked production Python modules under `ui/easydiffusion`, `ui/plugins`, and `training`. These are remaining native-port work, not harmless packaging details.
- `easy-difusion-tmp` is scratch only. `easy-diffusion-offline` can preserve historical offline data. `easy-diffusion-cpp-main` is obsolete unless an exclusive file is corroborated against active main, Theory, or a current requirement.

## Existing Cosmo evidence to preserve

- `cosmopolitan/PORTING.md` defines the whole-application contract: one executable, embedded UI resources, native server/API ownership, shared GGML, and no deployed Python runtime for supported native features.
- `cosmopolitan/docs/application-api.md` records current route/API inventory and remaining Python replacement work. Native parity must be route-by-route and behavior-by-behavior, not guessed from similar route names.
- `COSMOPOLITAN_INFERENCE_VALIDATION.md`, `COSMOPOLITAN_NATIVE_VULKAN_VALIDATION.md`, and `COSMOPOLITAN_WEBGPU_VALIDATION.md` prove specific CPU/software-WebGPU/native-provider milestones. They do not prove full API parity, all-GPU support, full training, or physical GPU acceleration across vendors.
- The Civitai image-v6 regression has been corrected in active main and Cosmo. The native port must preserve the new model-scoped behavior: image browsing searches model locations first, then lists images for a selected `modelId`/`modelVersionId`; standalone global `images_v6` search must not return.

## Implementation tracks

1. **Lock active-main parity as a contract.**
   - Generate a route, command, UI-action, plugin, model-tool, training, and configuration inventory from active main and Cosmo.
   - Classify each item as native-complete, Python-backed fallback, missing, intentionally unsupported, or duplicate/obsolete.
   - Preserve request schemas, response schemas, progress behavior, cancellation behavior, file outputs, settings persistence, errors, and privacy/logging behavior.
   - Keep the Civitai model-scoped image flow from `docs/todo/20-civitai-huggingface-ui.md` visible in the native route tests.

2. **Remove the constant deployed Python dependency.**
   - Replace `ui/easydiffusion` server modules, `ui/plugins/server` service modules, and `training` service entry points with native handlers or explicit temporary fallback boundaries.
   - Do not paper over missing native behavior with broad runtime fixers, auto-repair hooks, or silent Python subprocess fallbacks.
   - Build-time Python generators are acceptable only when their outputs and hashes are reproducible and the deployed supported feature no longer requires Python.

3. **Selectively redesign from Theory instead of wholesale copying it.**
   - Review Theory `source/DiffUser.cpp`, `source/INFERENCE.cpp`, `source/RestAPI.cpp`, `source/UI.cpp`, `source/API.test`, `Audit`, and backend documentation as candidate architecture/test material.
   - Treat Theory as a design/reference source with known quality issues. Port only concepts that survive current source review, compile review, and route/API contract tests.
   - Reconcile Theory backend matrices with existing Cosmo TODOs for CUDA, ROCm/HIP, oneAPI/SYCL, OpenVINO, OpenCL, Vulkan/WebGPU, DirectML/D3D12, JetPack, Windows, and aarch64.

4. **Complete native GPU and OS coverage by capability, not labels.**
   - Maintain CPU as the baseline supported backend.
   - Add or preserve validated native paths for NVIDIA CUDA, AMD ROCm/HIP where viable, Intel oneAPI/SYCL, OpenVINO, Vulkan/WebGPU, OpenCL, DirectML/D3D12, and Jetson-specific CUDA stacks where the existing TODOs require them.
   - Classify OpenGL compute only if a real supported path is chosen; do not keep it as a checkbox if it is not viable.
   - Every backend needs capability detection, unsupported-feature errors, fallback policy, model-loading behavior, memory limits, and a validation artifact that proves execution rather than enumeration.

5. **Turn WebGPU from “works slowly” into an optimized route.**
   - Keep the existing WebGPU evidence classified as execution proof through software Vulkan/lavapipe or native-provider discovery, not as a performance win.
   - Add controlled CPU versus WebGPU benchmarks with identical model, prompt, steps, scheduler, resolution, thread settings, and device policy.
   - Measure and reduce graph splits, CPU fallback fraction, readbacks, submission overhead, shader compilation cost, and memory movement.
   - Validate hardware WebGPU/native Vulkan on real discrete or integrated GPUs separately from embedded software Vulkan.

6. **Finish training, conversion, model tools, and plugin parity.**
   - Preserve native SD 1.5 LoRA training behavior while tracking SDXL/Anima, embeddings, optimizer options, resume, precision, sampling, and export gaps.
   - Port model listing, hashing, import/install, merge, LoRA metadata/trigger extraction, Hugging Face/Civitai download provenance, WD14 tagging, image tools, Perchance services, plugin controls, gallery/files behavior, kiosk mode, and diagnostics.
   - Keep LibTorch, ONNX Runtime, Transformers/tokenizer, GGUF conversion, and model-family dependencies explicit until each is either ported or intentionally unsupported.

7. **Package and validate one native artifact.**
   - Build one application artifact whose bytes are unchanged between supported Windows and Linux validation jobs.
   - Confirm UI resources, plugins, licenses, configuration templates, small validation fixtures, and native handlers are embedded or deployed as documented data.
   - Keep large user models, datasets, outputs, and user configuration external.

## Validation

- Static audit: run `python scripts/audit_cosmo_parity.py --format markdown` to refresh the non-runtime file, route, Python-surface, and Civitai image-flow inventory. This command deliberately records GPU execution as deferred and does not start the application.
- File parity: regenerate active-main/Theory/Cosmo tracked-file comparisons and investigate every active-main path missing from Cosmo.
- API parity: compare Python route declarations, native route declarations, plugin endpoints, generated OpenAPI or equivalent schema, and UI call sites; test both successful and failure responses.
- UI parity: run static UI integration tests and browser-based smoke tests for Generate, model browser, files/gallery, plugin controls, training status, settings, and cancellation where browser tooling is available.
- Civitai regression: verify no active `images_v6` route is used and that image browsing requires selected model context.
- Runtime parity: run CPU image generation, CPU HTTP generation/progress/cancellation, WebGPU tensor probes, WebGPU image generation, native-provider device tests, and backend-specific tests on actual matching hardware.
- Training parity: run at least one supported SD 1.5 native training recipe end to end and explicit rejection tests for unsupported training options.
- Packaging: run clean validation from the packaged artifact rather than the source tree, on Windows and Linux, with a fresh config/output directory.
- Documentation: update `cosmopolitan/PORTING.md`, `cosmopolitan/docs/application-api.md`, validation records, and TODO links whenever a track moves from fallback/missing to native-complete.

## Complete when

- Cosmo has no uninvestigated active-main parity gaps and every intentionally excluded active-main behavior has a documented reason.
- Supported UI/API features run through the one native application without requiring a deployed Python runtime.
- Theory-derived architecture or tests have either been ported with verification or explicitly rejected as obsolete/low-quality with a replacement plan.
- CPU, WebGPU/Vulkan, and each claimed vendor/backend route have execution evidence on the appropriate OS/hardware class.
- WebGPU has measured performance work and is no longer represented merely by slow software-driver proof.
- Training, online model services, model tools, plugins, diagnostics, configuration, and packaging have route-level/native validation.
- The final validation package identifies the exact source commit, artifact hash, model fixtures, hardware/driver matrix, commands, reports, and known unsupported features.
