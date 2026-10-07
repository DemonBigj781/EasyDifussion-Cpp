# Cosmopolitan native Vulkan and persistent configuration validation

**Validated on 2026-10-07.** Clean application source: [`1730424270e35ff4a06bd978654c9990cfd5777e`](https://github.com/DemonBigj781/EasyDifussion-Cpp/commit/1730424270e35ff4a06bd978654c9990cfd5777e). All four jobs in [workflow run 37568302821](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37568302821) completed successfully. The downloaded artifacts also passed a separate, offline inspection of **22 evidence groups**.

The executable now includes native Windows/Linux Vulkan discovery and device selection alongside the embedded Mesa software driver. The native Linux path completed actual GGML graphs, trained text inference and an HTTP diffusion image request through the host Vulkan loader. Persistent configuration is generated, saved and consumed by the CLI and native server; Settings and GPU Config use that same store.

**Physical GPU inference remains unvalidated.** The available compute adapters were CPU software implementations. The Windows native-driver check verified real DLL discovery and explicit rejection of an unsupported adapter; it did not execute native Windows GPU kernels. These distinctions are part of the result, not optional qualifications. The [machine-readable record](COSMOPOLITAN_NATIVE_VULKAN_VALIDATION.json) contains exact identities, counters, model provenance and report hashes.

## Exact application and download

Download `easy-diffusion-cosmopolitan-x86_64` from the [completed run](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37568302821). The ZIP includes the application, an optional explicit Linux loader, manifests and verification tools. The same `easy-diffusion.exe` is used on both operating systems.

| Item | Verified value |
| --- | --- |
| Application source | `1730424270e35ff4a06bd978654c9990cfd5777e`, `dirty=false` |
| Application size | 190,491,826 bytes |
| Application SHA-256 | `7dd513293f53bc39bd7c92e3aaff0522c89f0d543c3390b4dd95e2f590b507b9` |
| Application ZIP artifact ID | `11459728047` |
| ZIP size | 95,784,804 bytes |
| ZIP SHA-256 | `b223ad62158881434e1d87684282a0b73d99be3987ea46245d88a54386e62157` |
| Embedded resources | 850, with manifest hashes verified |
| Shared symbols | One definition of each audited GGML/model/backend entry point; no legacy versioned GGML copies |

The [build manifest](cosmopolitan/docs/native-vulkan-validation/ci/application/BUILD.json), [symbol audit](cosmopolitan/docs/native-vulkan-validation/ci/application/SYMBOLS.json) and [artifact provenance](cosmopolitan/docs/native-vulkan-validation/ci/ARCHIVE.json) bind these values to the tested application. This documentation is published in a later commit; the application was built from the clean source above. A documentation-only commit does not change the tested executable identity.

## Changes that address software-only detection

The [provider manager](cosmopolitan/backend/wgpu_devices.cpp) now enumerates native and embedded Vulkan providers separately, records actual adapter classification, and retains ownership for each device's context and buffers. An exact selector reaches the shared GGML backend used by image and llama inference. The mixed-provider check executes real graphs on both providers in one process.

The provider policies are:

| Policy | Behavior |
| --- | --- |
| `auto` | Prefer a compatible discrete or integrated GPU; otherwise select embedded software Vulkan. |
| `native` with `device=auto` | Require compatible physical hardware through the operating system's Vulkan loader. Fail when it is unavailable. |
| `native` with an explicit software selector | Allow an intentional host software-driver diagnostic, reported as software. |
| `embedded` | Use the bundled Mesa software driver. |

The `webgpu-device-test --require-hardware` diagnostic requires a native discrete/integrated GPU, rejects software and unknown adapter types, and executes four tensor graphs with twelve scalar comparisons. Device enumeration by itself does not pass this diagnostic. The backend currently requires `ShaderF16`, including for its F32 matrix kernels; incompatible adapters are listed with their reason. Unknown VRAM remains unknown, and adapter selectors are not presented as invented stable UUIDs.

Persistent `easy-diffusion.json` replaces the portable application's previous transient settings behavior. Defaults, saved values and temporary CLI overrides have explicit precedence. Atomic saves, strict schema validation, external-edit detection and restart requirements are documented in [CONFIGURATION.md](cosmopolitan/CONFIGURATION.md). Settings and Generate use actual native device records and keep software or unknown adapters accurately labeled.

Native Linux execution also required a threading correction. Cosmopolitan's native-library helper establishes host TLS on the original main thread. Both the outer HTTP inference closure and the diffusion loader's internal GPU uploads now execute there under Linux native/auto policy. Explicit embedded operation and Windows retain their worker behavior. The [backend contract](cosmopolitan/backend/README.md) describes this constraint, the host compiler prerequisite and the bounded native adapter-enumeration patch.

## Completed validation

| Check | Linux | Windows | Scope |
| --- | --- | --- | --- |
| Identical application bytes | Passed | Passed | x86-64 APE/PE; hashes checked before and after execution. |
| Embedded WebGPU graphs, trained text and shared storage | Passed | Passed | Real software-driver computation and scalar/reference checks. |
| CPU SD1.5 image CLI | Passed | Passed | 256×256, four steps, complete checkpoint to valid PNG. |
| Embedded WebGPU SD1.5 image CLI | Passed | Passed | 256×256, two steps, positive denoising counters and valid PNG. |
| CPU HTTP image lifecycle | Passed | Passed | Generation, responsive progress, request ownership and early cancellation. |
| Configuration HTTP save and actual restart | Passed | Passed | Saved settings become effective after restart; live options persist. |
| Configuration probe | 38 checks passed | 38 checks passed | Production configuration module; same probe bytes. |
| Device policy probe | Eight cases passed | Eight cases passed | Mock provider factory; separate from driver execution evidence. |
| Native Vulkan graphs and trained text | Passed | Not established | Linux host Mesa software adapter; real scalar checks and 16 reference tokens. |
| Native WebGPU HTTP image inference | Passed | Not established | Linux host Mesa software adapter; original-main execution. |
| Native DLL discovery and missing-feature rejection | Not this gate | Passed | Installed Chrome Vulkan/SwiftShader DLLs; no native compute claim. |
| Physical GPU inference | Not tested | Not tested | No suitable physical GPU was available. |

The [inspection report](cosmopolitan/docs/native-vulkan-validation/ci/inspection.json) links these checks to retained reports, logs and PNGs. Its 22 groups are an offline verification of completed CI evidence, not 22 additional executions. The [original build log](cosmopolitan/docs/native-vulkan-validation/ci/build-job.log) also records the Generate DOM mock and eight Settings UI scenario groups. Served-script hashes and DOM mocks do not establish real browser rendering or automation.

Linux's empty-filesystem runtime test covers the embedded backend, trained text fixture, server and UI without host libraries. Full diffusion tests ran separately with external model weights on the host. The shell bootstrap is another separate gate: `/bin/sh` and host utilities temporarily extract the bundled APE loader while preserving application bytes. Windows runs the PE directly.

### Full-model results and counters

All diffusion runs used the pinned SD1.5 Q4_0 checkpoint: **1,747,190,784 bytes**, SHA-256 `c2f6e92f9d08d69cc673a1003528ac8199274b3c0eaec88d5fbefe5af67bd42b`. The [model pin](cosmopolitan/docs/native-vulkan-validation/ci/application/PIN.json) records its repository, revision, license and complete-component requirements. The checkpoint is external to the application and is not committed to this repository.

| Completed workload | Elapsed seconds | Result |
| --- | ---: | --- |
| Linux CPU CLI, four steps | 347.844 | Valid PNG, exit 0 |
| Windows CPU CLI, four steps | 339.344 | Same PNG bytes as Linux, exit 0 |
| Linux embedded WebGPU CLI, two steps | 558.800 | Valid PNG, exit 0 |
| Windows embedded WebGPU CLI, two steps | 642.047 | Same PNG bytes as Linux, exit 0 |
| Linux CPU HTTP lifecycle suite | 359.195 | Four-step image plus separate early-cancellation request |
| Windows CPU HTTP lifecycle suite | 403.032 | Same image bytes; early cancellation passed |
| Linux native software WebGPU HTTP suite | 1,141.647 | Two-step image; valid PNG and successful execution marker |
| Local corrected native software CLI | 636.492 | Exact previous failing command completed, exit 0 |

These are recorded execution times, not controlled performance benchmarks. The API suites include lifecycle checks, and local/CI hosts and prompts differ. Two or four steps establish pipeline completion; image quality was not assessed.

The [Linux native HTTP report](cosmopolitan/docs/native-vulkan-validation/ci/linux-native/results-native-inference-api/report.json) records **287 graphs, 1,156 submissions, 6,061 dispatches, 1,608 matrix dispatches and 855 readbacks**, with `provider=native`, `software=1`, adapter type `3`, and one native loader open. Those counters cover the full inference closure, including model work. The native adapter was llvmpipe with LLVM 20.1.2, supplied by the host Mesa package `25.2.8-0ubuntu0.24.04.4`.

Both embedded CLI runs and the local native CLI recorded a separate interval between completed denoising steps: **126 graphs, 498 submissions, 2,582 dispatches, 692 matrix dispatches and 370 readbacks**. This interval excludes the first step, CLIP and VAE work. CPU fallback remains allowed in model inference, and its fraction was not measured. Positive WebGPU counters establish execution without claiming universal offload.

The native graph test performs four F32/Q4_0 graphs and twelve scalar comparisons without scheduler fallback. The [mixed-provider report](cosmopolitan/docs/native-vulkan-validation/ci/linux-native/results-dual-provider/report.json) repeats those graphs on both native and embedded devices within one process. Its later llama invocations are separate processes; provider coexistence does not establish simultaneous model sharding across multiple physical GPUs.

The [Windows DLL report](cosmopolitan/docs/native-vulkan-validation/ci/windows/results-windows-native-discovery/report.json) identifies installed Chrome `154.0.8037.98`, actual DLL hashes and `SwiftShader Device (Subzero)`. The adapter is rejected because the compiled kernels require `ShaderF16`. DLL bytes were hashed on the runner and are not included in the evidence archive. This successful rejection is not counted as Windows native GPU inference.

## The native diffusion failure and its correction

The first native build, source `9ff6b99`, passed small graph/text checks but failed real image loading with SIGSEGV. A local direct command reproduced the failure on that exact artifact. The [development archive](cosmopolitan/docs/native-vulkan-validation/development/README.md) preserves the failed report and a symbolized call stack from the diffusion loader's internal `std::thread` into WebGPU queue writes and native Vulkan buffer allocation. The top host-library frame remains unresolved.

[Patch 0012](cosmopolitan/app-patches/0012-native-model-loader-main-thread.patch) executes that loader body inline for Linux native/auto policy, preserves CPU-only conversion workers through an explicit host-only contract, and rejects wrong-thread native entry before work. Fourteen main-executor checks and nine model-loader checks passed; a separate controlled legacy worker reproduced the host-TLS fault.

The corrected CI application then completed the same direct image command locally and the full native HTTP gate in CI. The [local archive](cosmopolitan/docs/native-vulkan-validation/local/README.md) records one actual corrected app/model run, unchanged application/model hashes, both denoising callbacks, a valid PNG and normal exit. Only the loader, application and output PNG argv paths changed from the previous command; transient working/environment paths differed. A preliminary permission rejection never launched the application and is retained separately.

## Select and test your physical GPU

Windows requires a compatible installed Vulkan graphics driver providing `vulkan-1.dll`. Linux requires `libvulkan.so.1`, a compatible driver and a working host C compiler on `PATH` for Cosmopolitan 4.0.2's first native-library-helper creation. CPU and embedded software operation do not require that native compiler/driver path. No display server is required.

For a **new** configuration on Windows PowerShell:

```powershell
.\easy-diffusion.exe config init --config .\gpu.json --backend webgpu --provider native --device auto
.\easy-diffusion.exe devices --config .\gpu.json
.\easy-diffusion.exe webgpu-device-test --config .\gpu.json --device auto --require-hardware
.\easy-diffusion.exe llama --config .\gpu.json --prompt "Once upon a time" --tokens 16 --threads 1 --report-tokens
.\easy-diffusion.exe sdkit --config .\gpu.json --ckpt-dir .\models\checkpoints --port 8188
```

Run the hardware diagnostic first and proceed to model inference after it passes. The text command uses the embedded small trained fixture by default. Image inference requires a compatible complete checkpoint in the model directory. The server UI opens at `http://127.0.0.1:8188/`.

On Linux, use the same file with this executable prefix:

```sh
/bin/sh ./easy-diffusion.exe config init --config ./gpu.json --backend webgpu --provider native --device auto
/bin/sh ./easy-diffusion.exe devices --config ./gpu.json
/bin/sh ./easy-diffusion.exe webgpu-device-test --config ./gpu.json --device auto --require-hardware
/bin/sh ./easy-diffusion.exe llama --config ./gpu.json --prompt "Once upon a time" --tokens 16 --threads 1 --report-tokens
/bin/sh ./easy-diffusion.exe sdkit --config ./gpu.json --ckpt-dir ./models/checkpoints --port 8188
```

`config init` refuses to overwrite an existing file. For an existing configuration, inspect it with `config show`, edit it or use Settings, validate it with `config validate`, and restart the server after changing startup compute settings. CLI overrides are temporary. To persist a specific device, save the exact selector shown under the same provider policy in `compute.device`; registry numbering can change when devices or providers change.

If only llvmpipe is listed, physical GPU acceleration has not been established. A missing driver, inaccessible adapter or required-feature rejection must be resolved from the reported native-device reason. Selecting software or renaming its label does not provide GPU acceleration. Explicit software operation remains available with `--backend webgpu --provider embedded --device auto`.

## Remaining limits and evidence preservation

This milestone validates the tested x86-64 Windows/Linux inference paths and configuration behavior. It does not establish macOS/ARM portability, arbitrary GPU/vendor coverage, arbitrary checkpoint support, simultaneous multi-GPU model parallelism, full training or LibTorch parity, or complete replacement of every Python feature. Existing C++ and Rust dependencies remain linked; the application is not a pure-C implementation.

Native Linux HTTP cancellation was deliberately not exercised in the slow software-driver gate; CPU HTTP cancellation passed on both hosts. Embedded-provider WebGPU HTTP image generation and native Windows WebGPU image generation remain separate unvalidated paths. Native Linux/auto auxiliary HTTP upscaling remains explicitly unavailable until its host-library calls can follow the same safe thread contract.

The [CI archive](cosmopolitan/docs/native-vulkan-validation/ci/README.md) inventories 190 files totaling 2,857,845 bytes. Download ZIP sizes and SHA-256 digests were checked against GitHub metadata, selected archived members were compared with their original ZIP bytes, and original line endings are preserved. Its README and inventory are outside that inventory. The separate [development archive](cosmopolitan/docs/native-vulkan-validation/development/README.md) inventories 37 files; the [local corrected CLI archive](cosmopolitan/docs/native-vulkan-validation/local/README.md) inventories nine. Executables, loaders, debug binaries, models, DLLs and large dependency archives are excluded from these Git evidence directories.

The archived inspector can recheck the original five extracted artifacts against the matching source checkout's parsers. It never starts the application, loads a model or accesses the network. See the CI archive for the exact command. The [historical software inference record](COSMOPOLITAN_INFERENCE_VALIDATION.md) retains its original artifact identity and is not retroactively converted into native-driver or physical GPU evidence.
