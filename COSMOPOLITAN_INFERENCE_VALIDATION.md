# Cosmopolitan inference: one application verified on Windows and Linux

**One unchanged executable passed trained text and image inference on native Windows and Linux.** The completed run also passed CPU native HTTP image generation, responsive progress and early cancellation. WebGPU here uses the embedded Mesa/LLVM software implementation on the CPU; this is not physical GPU validation.

[Run 37551341258](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37551341258) completed successfully from source `6898a4e564c3363d1692f0aed1a2da0395a17a77` with `dirty=false`. The build job produced the application once; the runtime jobs downloaded those same bytes. The original metadata, independent package review, byte inventories and nine complete reports are retained below.

## Final application and artifact identity

| Field | Verified observation |
| --- | --- |
| Application | `easy-diffusion.exe`, 190,274,689 bytes |
| Application SHA-256 | `ad40cdabb2658bd064473012c32ac8f79bb2ad65557528707930529a888a9516` |
| Source / verifier commit | `6898a4e564c3363d1692f0aed1a2da0395a17a77`; clean source |
| Build job | [112567380884](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37551341258/job/112567380884), completed / success |
| Linux job | [112572262380](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37551341258/job/112572262380), completed / success |
| Windows job | [112572262321](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37551341258/job/112572262321), completed / success |
| BUILD metadata | [BUILD.json](cosmopolitan/docs/inference-validation/ci/application/BUILD.json), SHA-256 `48b0f64dfc54f1f57048f97d2d2feb80f82195f62eaf32761774aaa51b6f5367` |
| Shared-symbol audit | [SYMBOLS.json](cosmopolitan/docs/inference-validation/ci/application/SYMBOLS.json), SHA-256 `29826943a1f29615b4e2506e70d11767297d151e63027e1dd6b48180120aad6b` |
| Static dependency metadata | [LINK.json](cosmopolitan/docs/inference-validation/ci/application/software-webgpu/LINK.json), SHA-256 `dde3df9f8caa8a5a7eaa170c5c32deae472bc95aa4935b213973c45b6e571282` |
| Evidence inspection | [inspection.json](cosmopolitan/docs/inference-validation/ci/inspection.json), nine groups; actual binary and embedded resources rehashed |
| Complete file inventory | [FILES.json](cosmopolitan/docs/inference-validation/ci/FILES.json), 111 files / 2,640,242 bytes, SHA-256 `50bcb06c28bfd024006d44529c77238b31487aa79bf8de1d1fc0e0e10d4927b2` |

| Downloaded artifact | ID | ZIP bytes | Verified ZIP SHA-256 |
| --- | --- | ---: | --- |
| [easy-diffusion-cosmopolitan-x86_64](https://api.github.com/repos/DemonBigj781/EasyDifussion-Cpp/actions/artifacts/11453436083) | 11453436083 | 94,505,922 | `728373aaab3ca3208813c5bd0832a95cafe319f336ababf4d47e95732d890e50` |
| [cosmopolitan-linux-results](https://api.github.com/repos/DemonBigj781/EasyDifussion-Cpp/actions/artifacts/11453418678) | 11453418678 | 474,031 | `44d647c385981c2e92a1eaa36af0f37314d4bd2daa03960873a18634b53a39fd` |
| [cosmopolitan-windows-results](https://api.github.com/repos/DemonBigj781/EasyDifussion-Cpp/actions/artifacts/11454412392) | 11454412392 | 466,273 | `10e7dcac20654c19199a230225d6325380ca7d7ed132da211a876c3898e9efec` |

ZIP hashes identify transport archives; the executable hash identifies the one application file. Artifact expiry metadata is preserved in the JSON record. This verifies one produced artifact across operating systems, not independently reproducible builds.

## Same-file result matrix

| Host and gate | Observed result | Raw report |
| --- | --- | --- |
| Linux embedded runtime | PASS; 10 command/HTTP groups; embedded self-test 2.773 s | [report.json](cosmopolitan/docs/inference-validation/ci/linux/results-linux-isolated/report.json) |
| Linux shell bootstrap | PASS; 2.672 s; single-file shell launch with transient embedded-loader extraction | [report.json](cosmopolitan/docs/inference-validation/ci/linux/results-linux-bootstrap/report.json) |
| Linux CPU image | PASS; 346.935 s; 256 × 256, 4 steps, decoded PNG | [report.json](cosmopolitan/docs/inference-validation/ci/linux/results-inference-cpu/report.json) |
| Linux software WebGPU image | PASS; 557.356 s; 256 × 256, 2 steps, decoded PNG | [report.json](cosmopolitan/docs/inference-validation/ci/linux/results-inference-webgpu/report.json) |
| Linux CPU native API | PASS; 355.026 s; four-step PNG and eight-step request cancelled at step 1 | [report.json](cosmopolitan/docs/inference-validation/ci/linux/results-inference-api/report.json) |
| Windows embedded runtime | PASS; 10 command/HTTP groups; embedded self-test 4.062 s | [report.json](cosmopolitan/docs/inference-validation/ci/windows/results-windows/report.json) |
| Windows CPU image | PASS; 406.203 s; 256 × 256, 4 steps, decoded PNG | [report.json](cosmopolitan/docs/inference-validation/ci/windows/results-inference-cpu/report.json) |
| Windows software WebGPU image | PASS; 603.265 s; 256 × 256, 2 steps, decoded PNG | [report.json](cosmopolitan/docs/inference-validation/ci/windows/results-inference-webgpu/report.json) |
| Windows CPU native API | PASS; 405.781 s; four-step PNG and eight-step request cancelled at step 1 | [report.json](cosmopolitan/docs/inference-validation/ci/windows/results-inference-api/report.json) |

Linux embedded runtime checks use an empty chroot. Windows executes the same file as a native PE, without WSL. The large external-model image/API tests are not filesystem-isolated. Linux model runs use the explicit APE loader; the separate shell-bootstrap test uses host utilities and extracts its bundled loader temporarily.

## Actual image execution and output

Counts are graphs / submissions / dispatches / matrix dispatches / readbacks. Generation covers the complete engine call after context loading; sampling covers intervals between completed denoising steps and excludes the first step, CLIP and VAE.

| Host / backend | Steps | Generation counts | Sampling counts | Callbacks / intervals |
| --- | ---: | --- | --- | --- |
| Linux / cpu | 4 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 4 / 3 |
| Linux / webgpu | 2 | 287 / 1156 / 6061 / 1608 / 855 | 126 / 498 / 2582 / 692 / 370 | 2 / 1 |
| Windows / cpu | 4 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 4 / 3 |
| Windows / webgpu | 2 | 287 / 1156 / 6061 / 1608 / 855 | 126 / 498 / 2582 / 692 / 370 | 2 / 1 |

Both WebGPU reports identify the embedded software adapter and zero native Vulkan loader opens. CPU reports contain zero WebGPU counts. Diffusion CPU fallback is permitted; its fraction is unmeasured. The few-step outputs demonstrate completed inference and PNG decoding, not a visual-quality benchmark or CPU/WebGPU pixel equivalence.

| Output | PNG bytes | PNG SHA-256 |
| --- | ---: | --- |
| [Linux CPU image](cosmopolitan/docs/inference-validation/ci/linux/results-inference-cpu/image.png) | 156,929 | `ff2ae616f92aa8c3b94d0103711b3d6d756df87983a4a59b13f6176f682fc5dd` |
| [Linux software WebGPU image](cosmopolitan/docs/inference-validation/ci/linux/results-inference-webgpu/image.png) | 153,959 | `484325f78b9190ef0cf4f3de15fd35ef4bdc4cb1bf0fdc3fe75db17a4105c3ca` |
| [Linux CPU native API](cosmopolitan/docs/inference-validation/ci/linux/results-inference-api/api-image.png) | 120,155 | `de9199ded8368659705badde3e529dd593396c5da87d50e6d362d4b8edd47c39` |
| [Windows CPU image](cosmopolitan/docs/inference-validation/ci/windows/results-inference-cpu/image.png) | 156,929 | `ff2ae616f92aa8c3b94d0103711b3d6d756df87983a4a59b13f6176f682fc5dd` |
| [Windows software WebGPU image](cosmopolitan/docs/inference-validation/ci/windows/results-inference-webgpu/image.png) | 153,959 | `484325f78b9190ef0cf4f3de15fd35ef4bdc4cb1bf0fdc3fe75db17a4105c3ca` |
| [Windows CPU native API](cosmopolitan/docs/inference-validation/ci/windows/results-inference-api/api-image.png) | 120,155 | `de9199ded8368659705badde3e529dd593396c5da87d50e6d362d4b8edd47c39` |

Per-OS commands, platform strings, all callback durations, PNG statistics, and reported memory measurements with their methods are preserved in full in the JSON observations and original reports. No memory estimate is substituted when a report has no measurement.

## CPU HTTP lifecycle on both hosts

| Host | Total seconds | Active polls / maximum seconds | Worker stack / guard bytes | Cancelled step | Cleanup status |
| --- | ---: | --- | --- | --- | --- |
| Linux | 355.026 | 32 / 0.045916 | 8388608 / 4096; 8388608 / 4096 | 1 of 8 | 0 |
| Windows | 405.781 | 32 / 0.172 | 8388608 / 4096; 8388608 / 4096 | 1 of 8 | 1 |

Each server stayed alive until cleanup. Linux cleanup exits 0 after SIGTERM. Windows cleanup exits 1 because CPython explicitly calls `TerminateProcess(handle, 1)`; this is forced harness cleanup, not an inference failure or a claim of graceful Unix-style shutdown. Both tests require a matched interrupt after sampling begins, completed/interrupted state before step eight, and release of the generation gate. Busy generation/options mutations and wrong/finished/duplicate task identities receive 409; invalid overrides receive 400 without persisting request options.

These API tests select CPU. They do not establish WebGPU HTTP inference. The Node DOM mock passed in the build and is retained in [build-job.log](cosmopolitan/docs/inference-validation/ci/build-job.log); it exercises actual UI scripts with mocked browser interfaces. Served-asset hashes and real native HTTP model generation are separate evidence; no real browser automation is claimed.

## Image and API evidence definitions

The external model pin is [GPUStack SD1.5 Q4_0](cosmopolitan/inference/PIN.json), repository revision `dcac270609fffc0ce7c7d41a3c0e752721859f7c`, 1,747,190,784 bytes, SHA-256 `c2f6e92f9d08d69cc673a1003528ac8199274b3c0eaec88d5fbefe5af67bd42b`. The checkpoint includes F16 CLIP and VAE and Q4_0 diffusion weights with some F16 tensors. It is trained model data, licensed under the model card's `creativeml-openrail-m` declaration, and remains external to the executable and Git archive.

Each CLI report retains the exact command, parameters, model-pin metadata, separate stdout/stderr, elapsed time, unchanged application/model hashes, PNG bytes/hash, decoded dimensions, CRC checks and nonconstant pixel statistics. Memory values are included only with the report's measurement method and validated process identity. Linux's sampled `/proc` high-water mark may miss a final peak; Windows RSS must not be inferred from a local Linux value.

For WebGPU, the verifier requires `provider=embedded`, a software adapter, zero native Vulkan loader opens and positive graph/submission/dispatch/matrix-dispatch/readback deltas in two separate windows:

- `IMAGE_GENERATION`: the complete generation call after model/context loading, including conditioning and image decoding work.
- `IMAGE_SAMPLING`: intervals between completed denoising callbacks, excluding the first denoiser step, CLIP and VAE, with previews disabled. The two-step CI run contains two monotonic completed callbacks and one positive interval.

CPU fallback is allowed for diffusion and its fraction is unmeasured. The counters establish actual WebGPU work, not universal offload. PNG statistics establish image pipeline completion, not visual quality or equality with CPU output. The final two-step CI counts above are distinct from the local four-step values below.

The API gate tests **CPU HTTP inference only**. It returns one decoded 256 × 256 PNG with its requested task ID, exposes active and completed progress, preserves persistent options while applying a request-local checkpoint override, rejects concurrent generation/options mutation with 409, and rejects wrong/finished interruption IDs and duplicate task IDs with 409. Invalid overrides produce useful 400 errors and release the request gate. A separate eight-step request reaches sampling before interruption, finishes with `interrupted=true` and `current_step < 8`, and returns HTTP 200 with no images. `--skip-cancel` does not satisfy this gate. The first real request remains active through **32 progress polls**, each using a new connection and a two-second timeout, with the observed latencies retained. This detects a blocked Crow I/O worker even if a few initial progress/conflict requests reach other available workers. The server remains alive until harness cleanup; expected cleanup is Linux status 0 or Windows forced-termination status 1, as described above. Both real requests report their actual worker stack through `pthread_getattr_np`, at least 8 MiB each, with the measured guard size retained. The verifier rejects missing or unexpected worker records.

Served Generate-page scripts match the embedded resource hashes and advertise native single-user capabilities. The build's `UI_DOM_MOCK PASS` comes from the actual scripts evaluated with a mocked DOM, fetch transport and timers. It covers request mapping, selected backend, task-specific cancellation, progress, failure handling and preview/download URLs. It is not browser automation or real model generation. This evidence is separate from real HTTP image generation and the sixteen-page embedded C++ rendering checks.

## Existing text inference and numerical gates remain mandatory

CPU and WebGPU llama self-tests load the embedded trained `stories260K.gguf`, tokenize `Once upon a time`, prefill and perform sixteen greedy decode steps with finite logits. The required IDs, captured independently using original llama.cpp and original GGML, are:

`432,383,286,261,376,298,315,421,395,317,426,338,401,396,267,337`

[Reference provenance](cosmopolitan/docs/llama-reference.json), both token-marker lines and the `WEBGPU_LLAMA_DECODE_EXECUTION steps=16` counters are retained. All five decode-only counts are positive after a synchronized prefill snapshot; total inference counters alone could be satisfied by prefill. Normal llama scheduling still includes CPU buffers/operations. The strict direct graph probe remains separate: it executes F32/Q4_0 matmul → bias → RMSNorm → SiLU → softmax with WebGPU buffers, no CPU fallback, and twelve independent scalar readback checks.

The full runtime verifier also requires six alias-placement cases and twelve numerical checks, CPU/shared-GGML/training-math self-tests, device/help commands, malformed image/trainer argument rejection, embedded pages and native HTTP routes. The bootstrap parser requires all success markers and explicitly validates llama/WebGPU tensor and decode output; the raw alias-case output is retained as well. Negative command tests preserve application exit code 2 separately from the exact native host status: Linux 2 and Cosmopolitan 4.0.2 Windows 512. Separate stdout/stderr avoids the known Windows aliased-file-offset capture issue.

Historical same-file text/tensor evidence remains in [COSMOPOLITAN_WEBGPU_VALIDATION.md](COSMOPOLITAN_WEBGPU_VALIDATION.md), run `37531113555`, application SHA-256 `7d74b7f67750f6c6397ed3da95b9c3803ee030d08c975791c651acefc912481a`. That earlier artifact is separate from the final diffusion-inference artifact verified above.

## Completed local evidence, separated by artifact

The [local archive](cosmopolitan/docs/inference-validation/local/README.md) retains original reports/logs/PNGs and byte hashes. It contains development evidence, separate from the final CI artifact.

| Local result | Artifact SHA-256 | Observed result |
| --- | --- | --- |
| CPU image, 20 steps, eight GGML threads | `9adf1c7a76a31ea7fab38e467efcd5c61a363b31ded434054ad91e5006972841` | 256 × 256, exit 0, 394.635 s; observed application RSS 2,292,277,248 bytes; coordinator manually recognized an apple; automated `quality_assessed=false` retained |
| Patched WebGPU image, four steps, two GGML threads, two software-driver workers | `8205af0327e65b37fb35fb6e3acf38bcacc5f8f1e4694b539f82e6e859463131` | 256 × 256, exit 0, 1,015.189 s; observed application RSS 3,164,565,504 bytes; abstract low-step output, no quality-success claim |
| Patched alias-placement regression | `8205af0327e65b37fb35fb6e3acf38bcacc5f8f1e4694b539f82e6e859463131` | Six cases, twelve scalar readback checks, all pass; actual WebGPU dispatch and zero native loader opens |
| Corrected negative alias regression | `5638d6f0d34866612d11293df6d4379761a14abd5a34991843201940b9c9f99a` | Expected exit 1 before unsafe dispatch with patch 0005 reversed in an isolated core object/archive |
| Isolated API stack diagnostic | `ecee0f02e2671faf4cb718065a0c453fe243a97a959706e18a3fd65029c0d198` | Diagnostic debug ELF, confirmed worker-stack overflow, exit 139; not a passing inference artifact |
| Scoped inference-worker unit | `3cb7c1c4d5244ba6f0c6ace9b57a17d28aa598e948883f91f5c1bc47105cbdeb` | Separate no-model ELF; four actual 8 MiB stacks, join/result, exceptions and OOM thread-local-state checks pass; metadata reconstruction limits retained |
| Synchronous worker development run | `681bf68fe23430d189f46f27dd5fefb2052c17ef4cbdedf4e1a4c0e0971beca4` | First real CPU HTTP image completed; second-request progress timed out before cancellation; complete verifier failed and harness killed the server |
| Controlled asynchronous HTTP probe | `3020146aa9df003d18f863c314f748d96bf9bf62e646b4fbc6f96ac84980c3af` | Actual routes/dispatcher/Crow linked; two-second synthetic work replacement; progress, disconnect and shutdown checks pass; no model or PNG |
| Completed asynchronous CPU API run | `12e68947a6bdb61aedc1c76f3075c84a50b5ba72c0e2941e0597df10010e0374` | Actual four-step PNG, 32 responsive polls, separate eight-step request cancelled at step 1, two measured 8 MiB stacks, clean cleanup exit 0; 254.557 s |

The CPU image and WebGPU image differ in application bytes, steps and thread counts; their elapsed times and images are not a controlled backend comparison. The CPU build metadata records dirty intermediate source `dba9c50ea66680a01472e68e64c8cc46d054bdf7`. Both successful image reports validate unchanged app/model hashes.

The local WebGPU generation counts were **539 graphs / 2,152 submissions / 11,225 dispatches / 2,992 matrix dispatches / 1,595 readbacks**. Its denoising-only intervals recorded **378 / 1,494 / 7,746 / 2,076 / 1,110**, with four completed callbacks and three intervals. CPU fallback was allowed, its fraction unmeasured, and the provider was embedded software lavapipe/LLVM with zero native loader opens.

The original WebGPU failure at RIP `0x18fcd69` came from CPU-backed in-place aliases assigned to WebGPU after unsupported group normalization. Interpreting their CPU storage as a WebGPU buffer context faulted during a shared-pointer increment. The general shared-GGML placement fix selects one operation/storage-compatible backend for a canonical owner and its in-place aliases before copy splitting, preserving preallocated storage and the caller's fallback policy. The corrected negative test and six patched cases validate that invariant; the retained tests exclude the earlier incorrect optional CPU-device-pointer predicate.

A separate HTTP CPU-inference failure is diagnosed by [the exact stack trace](cosmopolitan/docs/inference-validation/local/api-stack-fault/symbolization.json): the actual worker stack was **81,920 bytes** with a 4,096-byte guard. RSP reached the lower bound and a `push %r13` in recursive `ggml_visit_parents_graph` wrote eight bytes below it. Thirty-two captured returns resolve immediately after the recursive call with 64-byte frame spacing. This establishes stack exhaustion, not infinite recursion. The isolated diagnostic ELF is distinct from the packaged `8205…` application. The current source implements a scoped 8 MiB worker for model creation, generation and PNG work while preserving request locking, exceptions and thread-local OOM behavior. The worker queries and reports its actual stack size rather than relying only on the requested pthread attribute. The later real CPU API run below passes actual image generation and cancellation. That pass is separate from the stack diagnosis, the worker unit and the earlier CLI results; the final native Windows/Linux CI result is recorded separately above.

### Worker unit, failed synchronous development run and controlled HTTP probe

The [scoped-worker unit](cosmopolitan/docs/inference-validation/local/api-worker-unit/build-and-run.json) is a no-model Linux check. Four calls reported actual 8,388,608-byte stacks and 4,096-byte guards. It verified work on a distinct pthread, a 256 KiB written/read stack payload, result lifetime after join, preserved `invalid_argument`/`runtime_error` types and messages, and native logging's simulated OOM state transfer followed by reset. This is a simulated error-state check, not an allocation-failure test. Its command provenance explicitly states that the compile/link argv was reconstructed after execution; source and executable hashes were captured afterward, without a before/after executable hash comparison. The combined log contains complete stdout followed by stderr, not chronological interleaving.

The subsequent [synchronous worker run](cosmopolitan/docs/inference-validation/local/api-synchronous-worker/report.json) completed one actual CPU HTTP image and request-conflict/invalid-override checks but **failed the full API verifier**. During the second, eight-step request a progress poll timed out after 10.0139 seconds, before the harness could send cancellation. The server was alive at cleanup and was then killed after termination did not finish, producing cleanup status -9; this is not recorded as a spontaneous model crash. Both raw 8 MiB stack markers were present. Its first PNG success did not establish progress/cancellation responsiveness.

The [controlled asynchronous probe](cosmopolitan/docs/inference-validation/local/http-async-probe/run/report.json) exercised actual production routes, dispatcher and Crow lifecycle with a linked diagnostic wrapper that deliberately discarded the expensive model-work closure, waited two seconds and returned base64 text `probe`. No real model or PNG was generated. It passed 32 active progress polls with maximum duration 0.001740563999 seconds, busy generation/options rejection with 409, completion after client disconnect, and clean exit 0 after SIGTERM while the third synthetic task was active. Shutdown took 2.020151758 seconds; the whole probe took 6.515541692 seconds. The production packaged application `12e68947…` and protected build inputs stayed unchanged during the isolated relink. This diagnostic executable is separate from the later real-model API pass.

### Completed local real-model CPU API validation

The [passing API report](cosmopolitan/docs/inference-validation/local/api-async/report.json), [original logs](cosmopolitan/docs/inference-validation/local/api-async/server.stdout.log) and [build identity](cosmopolitan/docs/inference-validation/local/api-async/BUILD.json) establish the tested native request lifecycle on the actual packaged `12e68947…` application. It contains no synthetic model-work replacement. Total verifier time was **254.557 seconds**.

The first request returned one real 256 × 256 PNG after four CPU steps, using the trained pinned SD1.5 checkpoint, seed 42, CFG 7, Euler and discrete scheduling. PNG SHA-256 is `de9199ded8368659705badde3e529dd593396c5da87d50e6d362d4b8edd47c39`, 120,155 bytes. Independent decoding verified CRCs and nonconstant pixels; `quality_assessed=false` remains unchanged. This is a pipeline result, not an image-quality benchmark.

All **32 active progress polls** completed within their two-second per-call limit, with a maximum of **0.002233 seconds**. Busy generation/options, wrong/finished-task interruption and duplicate task IDs produced the expected 409 responses. Invalid overrides produced useful 400 errors, left persistent options unchanged and released the gate. The first task completed at step 4 of 4 with its matching request ID.

A separate eight-step request reached completed sampling step 1 before its matched interrupt was sent. The interrupt returned HTTP 200 and the generation response returned HTTP 200 with no images. Final progress was step 1 of 8, `completed=true`, `interrupted=true`; early cancellation and gate release were verified. The engine's error-level sampling messages immediately follow the matched interrupt and graph-cancellation message; they are retained as expected interruption output, not treated as an unhandled inference error.

Both real requests reported actual 8,388,608-byte stacks and 4,096-byte guards through `pthread_getattr_np`. The server remained alive through all checks and exited 0 during harness cleanup. Model and application hashes were unchanged. The report also verifies served Generate-script hashes and native single-user capabilities; `browser_execution_tested=false`. It does not measure application RSS or validate WebGPU HTTP inference.

The local API application is 190,270,314 bytes. Its original build metadata says source `c6b876179c88ad02c811d81b81fb08c79efc2cdf`, `dirty=true`; it must not be relabeled as a clean build of the subsequently published `6898a4e…` commit. The final CI build independently verifies the published implementation with the distinct clean-build hash and reports recorded above.

## Implementation and limits

The executable combines C dispatch, llama orchestration and the image CLI with the existing C++ model engines/shared GGML/server/UI renderer/LLVM and Rust wgpu-native. A narrow C++ boundary catches engine exceptions. Both llama and diffusion consume one authoritative GGML, and the image command generates an actual PNG through the existing engine. The asynchronous HTTP route copies request data into owned state and submits it to a single-active-job dispatcher away from Crow I/O threads; completion is posted back to the original I/O executor. Crow retains connection ownership through response completion, and shutdown closes admission, joins producers after I/O loops stop, then abandons undelivered responses without middleware or socket writes. The model work still runs on the explicitly measured 8 MiB worker, preserving exception and native OOM state handling. These lifecycle fixes are covered locally by the real CPU API run and the separate controlled disconnect/shutdown probe; routing other image/video endpoints through the dispatcher does not validate those model modes. No Python process performs inference. A normal browser consumes the served UI; the complete Python application service has not been replaced or validated route-for-route.

The current dependency pin is foundation revision `c2a7b3f8c871e44ae1e3fc8db36759c7b1829326`, wgpu-native 29.0.1.1, Mesa 25.2.8, LLVM 19.1.7, Cosmocc 4.0.2 and Rust nightly-2026-07-28. The final package review cross-checked these pins and patch hashes against the produced metadata. The preserved kernels require `ShaderF16`, even for F32 matmul staging. The pinned native C feature mapping does not expose standard subgroups; subgroup-dependent flash attention is rejected. Packed integer-dot kernels and timestamp GPU profiling remain disabled. This is CPU software compute through embedded Vulkan, not physical GPU acceleration or proof of coverage across GPU vendors or all x86-64 CPUs.

The native Generate form covers one text-to-image request at a time. Companion-model selection, LoRA, ControlNet, image-to-image, plugins and non-PNG generation are outside this form's tested scope. Cancellation during model loading is not implemented. No evidence here certifies full SD training, optimizer/model-family parity, production image quality, video inference, arbitrary checkpoint support, or every UI/API route. Existing partial-training semantics and strict no-fallback training behavior remain separate requirements.

## Retained evidence

[Original run](cosmopolitan/docs/inference-validation/ci/run.json), [jobs](cosmopolitan/docs/inference-validation/ci/jobs.json), [artifact listing](cosmopolitan/docs/inference-validation/ci/artifacts.json), [inspection](cosmopolitan/docs/inference-validation/ci/inspection.json) and [complete CI inventory](cosmopolitan/docs/inference-validation/ci/FILES.json) preserve the final evidence. The local development archive and earlier milestones remain distinct and retain their original hashes and failure classifications. Model weights, application/loader/debug binaries and static archives are not committed into this evidence archive.

The [machine-readable record](COSMOPOLITAN_INFERENCE_VALIDATION.json) includes every original report body, observed timings/counters, download verification, source identity and byte hashes. Preparing this publication record performed no application or model execution.
