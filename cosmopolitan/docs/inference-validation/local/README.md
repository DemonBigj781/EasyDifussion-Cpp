# Local diffusion inference and lifecycle evidence

This archive records completed local Linux runs of trained SD1.5 CPU image generation, four-step diffusion inference through embedded software WebGPU, and real CPU HTTP image generation with responsive progress and early cancellation. It also preserves the earlier failures and focused checks that led to these results: the in-place storage regression, native worker-stack diagnosis and unit, partial synchronous HTTP run, and controlled synthetic asynchronous HTTP probe.

These are development runs through an explicit APE ELF loader on different identified application artifacts. The WebGPU run proves actual denoising work and decoded PNG output with CPU fallback allowed; its abstract four-step image does not establish visual quality. The real API run proves the tested CPU request lifecycle and served assets, not browser execution or WebGPU HTTP inference. Windows diffusion inference, filesystem-isolated large-model inference, and final same-artifact CI inference validation remain pending. Final validation belongs in the later top-level inference record.

The copied reports, logs, PNG, and build metadata retain their original bytes, including recorded host paths. [FILES.json](FILES.json) lists their original repository-relative locations, lengths, and SHA-256 hashes. Its derived entries comprise two offline symbolization records and the explicitly reconstructed worker-unit provenance; the API diagnostic wrapper source and its exact captured build provenance are also retained. The archive contains no model weights, application executables, ELF loaders, debug binaries, object files, or static libraries. The local `.gitattributes` disables text conversion so checkout settings do not change archived hashes.

## Artifact identities

| Evidence | Application SHA-256 | Identity and scope |
| --- | --- | --- |
| CPU 20-step image and diagnostic WebGPU failure | `9adf1c7a76a31ea7fab38e467efcd5c61a363b31ded434054ad91e5006972841` | Intermediate 190,222,324-byte APE; [BUILD.json](cpu-quality/BUILD.json) records source commit `dba9c50ea66680a01472e68e64c8cc46d054bdf7` with `dirty: true`. This diagnostic build contained temporary fault-trace code, subsequently removed. It predates the alias-placement fix and differs from the final CI artifact. |
| Corrected negative scheduler regression | `5638d6f0d34866612d11293df6d4379761a14abd5a34991843201940b9c9f99a` | Isolated debug ELF linked with only patch 0005 reversed in a separate scheduler object/base archive. Its exact corrected test object and original-input integrity checks are recorded in [build.json](inplace-baseline-corrected/build.json) and [report.json](inplace-baseline-corrected/report.json). |
| Patched scheduler regression and completed four-step WebGPU image inference | `8205af0327e65b37fb35fb6e3acf38bcacc5f8f1e4694b539f82e6e859463131` | Intermediate packaged APE with patch 0005 and the corrected test predicate; [regression report](inplace-patched-validated/report.json) and [inference report](webgpu-fixed/report.json). This is a separate artifact from the CPU image run. |
| Isolated native HTTP CPU-inference stack diagnostic | `ecee0f02e2671faf4cb718065a0c453fe243a97a959706e18a3fd65029c0d198` | Separate debug ELF linked from the current application objects plus an opt-in `--wrap=generate_image` diagnostic. [Build provenance](api-stack-fault/build.json) records the unchanged `8205…` packaged application and exact compiler/link arguments. The wrapper is diagnostic-only, and this run failed. |
| Scoped worker CPU HTTP image, followed by failed progress/cancellation gate | `681bf68fe23430d189f46f27dd5fefb2052c17ef4cbdedf4e1a4c0e0971beca4` | Intermediate 190,248,333-byte APE; [BUILD.json](api-synchronous-worker/BUILD.json) records source `c6b876179c88ad02c811d81b81fb08c79efc2cdf`, `dirty: true`. [Report](api-synchronous-worker/report.json) retains overall `status: failed`; one first-request PNG succeeded. |
| Controlled asynchronous HTTP lifecycle probe | `3020146aa9df003d18f863c314f748d96bf9bf62e646b4fbc6f96ac84980c3af` | Separate diagnostic ELF with a two-second synthetic replacement for model work; [report](http-async-probe/run/report.json) explicitly records `real_model_called: false` and `real_png_generated: false`. The production packaged application remained `12e68947a6bdb61aedc1c76f3075c84a50b5ba72c0e2941e0597df10010e0374`. |
| Completed real CPU HTTP inference and lifecycle | `12e68947a6bdb61aedc1c76f3075c84a50b5ba72c0e2941e0597df10010e0374` | Intermediate 190,270,314-byte APE; [BUILD.json](api-async/BUILD.json) records source `c6b876179c88ad02c811d81b81fb08c79efc2cdf`, `dirty: true`. [Report](api-async/report.json) passes real image, active progress, task-specific early cancellation, gate release and served-asset checks. This local build is not the later clean CI artifact. |

The regression patch is [0005-preserve-inplace-storage-placement.patch](../../../shared-ggml/patches/0005-preserve-inplace-storage-placement.patch), SHA-256 `b9a996cf03a5ffff91d6a6d5ac8708f7d73a7fccbb41898ff396d818bd2b2f8d`. The negative and positive runs use the same corrected test object, SHA-256 `b55fc3332b6e73ccd7b46fe7afa06309434d85d9defd631507be994f0375b2fa`. They are intentionally different executables; this comparison isolates the scheduler change and is not a same-artifact cross-OS test.

## Real trained-model CPU image

The [CPU report](cpu-quality/report.json), [stdout](cpu-quality/inference.stdout.log), and [stderr](cpu-quality/inference.stderr.log) record successful generation from the trained SD1.5 Q4_0 checkpoint, with CLIP and VAE included. The external model is `gpustack/stable-diffusion-v1-5-GGUF` at revision `dcac270609fffc0ce7c7d41a3c0e752721859f7c`, file `stable-diffusion-v1-5-Q4_0.gguf`, 1,747,190,784 bytes, SHA-256 `c2f6e92f9d08d69cc673a1003528ac8199274b3c0eaec88d5fbefe5af67bd42b`. The report retains the exact model-pin metadata and `creativeml-openrail-m` license identifier. The model is not embedded in the application or copied into this archive.

| Parameter or observation | Recorded value |
| --- | --- |
| Prompt | `a photograph of a red apple on a wooden table, natural light` |
| Backend | CPU |
| Image | 256 × 256, RGB |
| Sampling | 20 steps, seed 42, `euler_a`, `discrete`, CFG 7 |
| GGML threads | 8 |
| Process elapsed time | 394.635 seconds |
| Generation-call time | 394.293350 seconds |
| Observed application RSS high-water mark | 2,292,277,248 bytes |
| Sampling progress | 20 monotonic completed callbacks; 19 intervals between completed steps |
| WebGPU generation and sampling counters | All zero |
| Exit status | 0; `IMAGE_INFERENCE PASS` |

Memory was sampled from Linux `/proc` every 100 ms after validating the application's PID namespace, parent, command, and executable mapping. The report records 3,901 validated samples. This is the application's observed `VmHWM`, not a process-tree sum or a guaranteed final peak; sampling may miss a later peak. A final unavailable sample after process exit is retained in the original report. This archive excludes the earlier four-step CPU report whose RSS measurement was invalid because the process identity was not correctly resolved.

The verifier decoded [image.png](cpu-quality/image.png), checked PNG chunk CRCs, dimensions, nonconstant color pixels, and statistics against the application's output marker. It found 120,655 PNG bytes, color range 0–255, mean 91.134129842, and standard deviation 81.003627927. PNG SHA-256 is `795aed477993d4d9187f40d6eebc2e7344b367ba11bd4a041e31aaf7a4d85e63`. Application and model hashes were unchanged after execution.

![CPU-generated red apple](cpu-quality/image.png)

The coordinating agent manually inspected this image and reported a recognizable apple. That observation is separate from automated validation: the original report deliberately retains `quality_assessed: false`. One prompt at 256 × 256 is not a production-quality assessment or a quality benchmark. The adapter announcement in stderr reflects application initialization; the CPU-selected run's zero WebGPU execution counters establish that it does not supply WebGPU image-generation evidence.

## Completed four-step embedded-software WebGPU inference

The [WebGPU report](webgpu-fixed/report.json), [stdout](webgpu-fixed/inference.stdout.log), and [stderr](webgpu-fixed/inference.stderr.log) record successful completion of the same trained SD1.5 checkpoint and prompt with the patched `8205…` application. This run used 256 × 256 output, four steps, seed 42, two GGML threads, CFG 7, `euler_a`, and `discrete`. The software driver had a separate worker limit of `LP_NUM_THREADS=2`. Process elapsed time was **1,015.189 seconds**; the generation call took 1,014.720626 seconds. Exit status was 0 and `IMAGE_INFERENCE PASS` was present.

The provider was `embedded`, the software adapter was `llvmpipe (LLVM 19.1.7, 256 bits)` with Mesa 25.2.8, and the native Vulkan loader-open count was zero. This path runs on the CPU through the embedded software Vulkan driver. Diffusion's CPU backend fallback was allowed and its fraction was not measured. The recorded counters demonstrate actual WebGPU work during both complete generation and the denoising-only interval; they do not claim that every operation ran through WebGPU.

| Counter | Complete generation call after model/context loading | Between completed denoising steps 1 and 4 |
| --- | ---: | ---: |
| Graphs | 539 | 378 |
| Queue submissions | 2,152 | 1,494 |
| Compute dispatches | 11,225 | 7,746 |
| Matrix-multiplication dispatches | 2,992 | 2,076 |
| Readbacks | 1,595 | 1,110 |

Four completed sampling callbacks were monotonic. The three measured intervals between them exclude the first denoiser step, CLIP, and VAE; previews were disabled. All five counters are positive in this denoising-only interval, so the evidence cannot be satisfied solely by model loading or CLIP/VAE work.

The observed application RSS high-water mark was **3,164,565,504 bytes**, based on 10,072 validated samples from the namespace-aware process-identity helper. The same 100 ms `/proc` sampling limitation described for the CPU run applies: this is an observed application `VmHWM`, not a guaranteed final peak or process-tree sum. Application and model SHA-256 values were unchanged after execution.

The verifier decoded [image.png](webgpu-fixed/image.png), validated chunk CRCs, dimensions, and nonconstant RGB pixels, and matched the output statistics. The PNG contains 153,820 bytes and has SHA-256 `c40765713f2cfa8674bf49721aa956e6407b3f3ad521f32ee9ff4cab2735b3f8`; its color range is 0–255, mean 77.983637492, and standard deviation 80.529103971. The report retains `low_step_plumbing: true` and `quality_assessed: false`. The four-step output is abstract and is retained as pipeline-completion evidence, not a successful visual-quality result.

![Four-step WebGPU plumbing output](webgpu-fixed/image.png)

The separately inspected 20-step CPU apple image used a different executable, eight GGML threads, and twenty steps. These runs are not a controlled backend speed or image-quality comparison. The earlier synchronous HTTP run below completed one image response but failed the later progress/cancellation gate. The subsequent [asynchronous CPU API run](api-async/report.json) separately passes the tested request lifecycle; CLI success alone does not establish that behavior.

## Corrected alias-placement regression

The [negative report](inplace-baseline-corrected/report.json), [stdout](inplace-baseline-corrected/stdout.log), and [stderr](inplace-baseline-corrected/stderr.log) record the expected failure before unsafe graph dispatch. The `GROUP_NORM` owner was assigned to CPU. Its in-place multiply, add, SiLU, and reshape aliases were assigned to WebGPU while retaining the same CPU buffer and data address. The run exited 1 after 0.064117 seconds and the report explicitly validates this failure kind and unchanged original build inputs.

This is the corrected negative test. It does not use the discarded predicate that required a CPU buffer type's optional device pointer to equal the CPU backend device. That optional pointer is null in this pinned implementation. The retained test checks actual backend assignment and compatible storage; it fails on `mul_inplace`, not on the valid CPU owner.

The [patched report](inplace-patched-validated/report.json), [stdout](inplace-patched-validated/stdout.log), and [stderr](inplace-patched-validated/stderr.log) record six cases and twelve scalar readback checks, all passing in 0.368911 seconds. Each case runs twice with different deterministic inputs:

| Case | Placement and storage checked | Per-case graphs / submissions / dispatches / matmuls / readbacks |
| --- | --- | --- |
| Scheduler-allocated CPU owner | CPU group normalization and in-place affine/SiLU aliases retain common CPU storage; the out-of-place matrix consumer executes through WebGPU | 1 / 4 / 1 / 1 / 3 |
| Preallocated CPU owner | The same alias invariant with an existing CPU allocation, followed by the WebGPU consumer | 1 / 4 / 1 / 1 / 3 |
| Preallocated WebGPU owner | SiLU and its alias stay on the existing WebGPU storage, followed by WebGPU matrix multiplication; CPU fallback disabled | 1 / 3 / 2 / 1 / 2 |

Every case reports `native_loader_opens=0`. The software adapter is `llvmpipe (LLVM 19.1.7, 256 bits)`, Mesa 25.2.8. Maximum activation error across the twelve checks is `1.1920929e-07`; maximum projection error is `4.47034836e-08`. The independent projection reference models the existing WebGPU kernel's F16 input staging. These small graphs exercise actual WebGPU dispatch and readback plus the mixed-backend alias invariant. They do not establish completion of an entire diffusion model or support for every operation.

## Diagnostic full-model WebGPU failure

The [diagnostic report](webgpu-fault/report.json), [stdout](webgpu-fault/stdout.log), and [stderr](webgpu-fault/stderr.log) retain the failed 256 × 256, four-step, two-thread attempt with the intermediate `9adf…` application. Model context creation reported success, but the process faulted before any completed sampling callback or successful output image. The recorded diagnostic exit status is 139 after 22.178546 seconds; no successful generation is inferred from model loading or adapter enumeration.

The temporary signal handler recorded SIGSEGV, RIP `0x18fcd69`, and RCX `0xbe3c80f5bd554e8c`. [symbolization.json](webgpu-fault/symbolization.json) preserves offline `addr2line` and `objdump` output against the exact saved intermediate debug file, SHA-256 `4be41d568ead3ac72c308919d59b27120cf78653338eee54c0dedfb395cf274d`. The address resolves to `ggml_webgpu_make_tensor_bind_group_entry`; the instruction is `lock addq $0x1,0x8(%rcx)`, a shared-pointer reference-count increment. Source inspection and the corrected negative regression identify the incompatible CPU storage/WebGPU placement that allowed a CPU buffer context to be interpreted as a WebGPU buffer context.

The debug binary was used only for symbolization and is not archived here. The signal trace establishes the failure location; the six passing regression cases establish the targeted placement fix. The later [completed WebGPU image report](webgpu-fixed/report.json) establishes local four-step full-model inference with actual denoising work and CPU fallback allowed. The later [CPU API report](api-async/report.json) separately passes the tested local HTTP lifecycle; Windows and final same-artifact CI inference verification remain separate gates.

## Confirmed native HTTP worker stack exhaustion

The isolated [API report](api-stack-fault/report.json), [server stdout](api-stack-fault/server.stdout.log), and [server stderr](api-stack-fault/server.stderr.log) preserve a failed native `POST /v1/sdapi/v1/txt2img` request. Checkpoint listing returned the indexed SD1.5 Q4_0 model, context initialization succeeded, and CLIP conditioning completed. The HTTP connection then closed before any image response. The request used CPU, 256 × 256, four steps, seed 42, Euler, discrete scheduling, and CFG 7. The process exited 139 after 3.770822 seconds; the diagnostic handler deliberately calls `_Exit(128 + SIGSEGV)` after recording the fault. This is failure evidence, not an API inference pass.

The [isolated C wrapper](api-stack-fault/inference-api-trace.c) intercepts `generate_image` after context creation. With `COSMO_API_TRACE_FAULT=1`, it queries the calling thread through `pthread_getattr_np`/`pthread_attr_getstack`, installs a fresh 64 KiB alternate signal stack on that thread, and records SIGSEGV registers with manual formatting and `write`. It does not change the normal worker stack size. Its [exact build provenance](api-stack-fault/build.json) retains compiler/link commands, wrapper-source SHA-256, and before/after integrity checks for the existing application inputs. No production source was changed to produce this diagnostic executable.

The captured worker stack spans `0x7fa139d1f000`–`0x7fa139d33000`: **81,920 bytes (80 KiB)**, with a reported 4,096-byte guard. At the fault, RSP equals its lower bound and the faulting address is `0x7fa139d1eff8`, **eight bytes below that bound**. The handler's recorded stack address lies within the separate 64 KiB alternate stack, confirming it could execute despite exhaustion of the normal worker stack.

[Offline symbolization](api-stack-fault/symbolization.json) against the exact `ecee…` diagnostic executable resolves RIP `0x4c2331a` to `ggml_visit_parents_graph`; disassembly identifies the faulting instruction as `push %r13`. All 32 captured frame-pointer returns are `0x4c233b3`, immediately after that function recursively calls itself, with 64-byte frame spacing. These observations establish stack exhaustion during recursive graph construction. The bounded trace does not establish an infinite recursion or provide the complete call chain. Source line numbers are unavailable in this optimized object; the exact symbol and instruction are retained.

At archival time, the packaged application's SHA-256 remained `8205af0327e65b37fb35fb6e3acf38bcacc5f8f1e4694b539f82e6e859463131`. The isolated diagnostic executable differs from that packaged APE and is not copied into this archive. A scoped worker-stack fix and a fresh passing HTTP inference/lifecycle run require separate evidence; this trace makes no claim that the API issue has already been resolved.


## Scoped inference-worker unit

The [exact test source](api-worker-unit/worker-test.cpp), [original combined output](api-worker-unit/combined.stdout-then-stderr.log), empty [compiler output](api-worker-unit/build.log), and [build/run provenance](api-worker-unit/build-and-run.json) record a separate **no-model** unit test of the production worker helper. The isolated ELF was compiled with Cosmocc 4.0.2 and run through the explicit APE ELF loader on Linux. Compiler and runtime exit statuses were both 0. Its SHA-256, measured during archival, is `3cb7c1c4d5244ba6f0c6ace9b57a17d28aa598e948883f91f5c1bc47105cbdeb`; its size is 10,111,916 bytes. The executable and loader are not archived.

Four successive worker calls each reported an actual **8,388,608-byte stack** and **4,096-byte guard** through `pthread_getattr_np`. The test checks that execution takes place on a different pthread, touches and verifies a 256 KiB volatile stack buffer, returns its result after joining, and preserves the original `std::invalid_argument` and `std::runtime_error` types and messages. It invokes the real native `sd_log_cb` with a deliberately simulated Vulkan out-of-memory message, verifies that the worker's thread-local error flag reaches the caller, then verifies that a subsequent worker clears that stale state. The error-colored OOM line in the log is intentional test input, not an observed allocation failure.

The output file preserves the original runner's **complete stdout followed by complete stderr**; it is not a chronological merge. Thus the summary appears before the four worker markers in this archived file. The build log is intentionally empty because the compiler emitted no stdout or stderr.

The provenance identifies the exact test, production helper, and scratch patched logging source/header bytes used by the compiler, and the archive retains those source snapshots under [sources](api-worker-unit/sources/). The scratch logging files were checked to match the production staged files at archival time. Source and executable hashes were captured after this passing run; there was no pre-run hash inventory or before/after executable integrity check for this unit. The compiler/link argv was **reconstructed after execution** from the recorded orchestration procedure and current unchanged `image_bridge.cpp` compilation flags. It was not printed or separately captured during the original compile. The metadata marks this distinction explicitly, retains the compiler wrapper and run invocation, and makes no claim of an independently captured original command.

This unit establishes the helper's stack allocation, synchronous lifetime, exception propagation, and native OOM state transfer. It does not exercise a model, the HTTP server, request cancellation, Windows, GPU execution, or injected pthread creation/join failures. A real API inference/lifecycle pass remains separate evidence and is not asserted here.


## First HTTP image completed; second-request progress/cancellation failed

The [unchanged report](api-synchronous-worker/report.json), [server stdout](api-synchronous-worker/server.stdout.log), [server stderr](api-synchronous-worker/server.stderr.log), [PNG](api-synchronous-worker/api-image.png), and [build metadata](api-synchronous-worker/BUILD.json) preserve the next local CPU API run with the scoped worker fix. **The complete verifier failed.** This record separates the successful first image and completed request checks from the failed second-request progress/cancellation gate.

The first `POST /v1/sdapi/v1/txt2img` returned HTTP 200 with one actual PNG and the matching requested task ID. It used the same pinned trained checkpoint, 256 × 256, four steps, seed 42, CFG 7, Euler and discrete scheduling. The HTTP request took 201.0163 seconds; the server logged generation in 200.83 seconds. The independently decoded PNG is 120,155 bytes, SHA-256 `de9199ded8368659705badde3e529dd593396c5da87d50e6d362d4b8edd47c39`, with valid chunk CRCs, nonconstant RGB pixels, range 0–255, mean 82.964162191 and standard deviation 66.812027913. The original `quality_assessed: false` remains unchanged; this is not a visual-quality result.

Before the first image completed, the harness observed an active request with step 0 and rejected concurrent image generation and options mutation with HTTP 409. A wrong-task interrupt also returned 409. After image completion, final progress recorded step 4 of 4, `completed=true`, `interrupted=false`; persistent options were unchanged. Finished-task interruption and reuse of the task ID returned 409. Three malformed/missing checkpoint overrides returned informative 400 errors and released the gate. Capability/kiosk checks and served Generate-script hashes passed, with `browser_execution_tested=false`. These are the completed first-request and request-validation checks, not proof that progress remained responsive throughout denoising.

The separate second request asked for eight steps so early cancellation could be tested. Initial progress queries returned step 0, then a `POST /v1/internal/progress` timed out after **10.0139 seconds**. The harness did not reach the sampling-observed state and **did not send a cancellation request for that second task**. There is no successful cancellation record. The report's `status` is `failed`, its error is `timed out`, and total verifier elapsed time is 225.05 seconds.

`server_exit_before_cleanup` is null: the server was still running when the harness began cleanup. After termination did not finish within the cleanup grace period, the harness killed it; `server_cleanup_exit_status` is **-9**. The second generation connection then closed without a response. The cleanup termination/kill must not be relabeled as a spontaneous inference crash. The stdout lines closing Crow I/O services belong to this cleanup sequence.

Both requests printed measured worker stacks of **8,388,608 bytes** and 4,096-byte guards through `pthread_getattr_np`. These raw markers demonstrate that the enlarged worker stack was used, but the failed run did not reach the verifier's final worker-record acceptance check. The report verifies unchanged application and model hashes. It does not provide a validated application RSS measurement.

The next change keeps the Crow I/O worker responsive while inference runs and the request retains its response/lifetime state. The controlled asynchronous probe below tests that route/dispatcher behavior with substituted work. It still requires a fresh complete real-model image/progress/cancellation run; the successful first PNG and synthetic probe are not a complete API inference pass, and this archive makes no Windows or CI claim for that change.


## Controlled asynchronous HTTP probe — no model or PNG generation

The [probe report](http-async-probe/run/report.json), [server stdout](http-async-probe/run/server.stdout.log), and [server stderr](http-async-probe/run/server.stderr.log) record a passing **controlled HTTP lifecycle probe**, not an image-inference test. The isolated executable links the actual production Server routes, dispatcher and Crow lifecycle but replaces `cosmo_inference_worker` with the archived [wrapper](http-async-probe/wrapper.cpp). That wrapper ignores the expensive work closure, delays for approximately two seconds, and returns `cHJvYmU=` (base64 for the text `probe`). It neither loads a model nor produces a PNG. The temporary `.gguf` file is an indexing placeholder, not model weights.

The [original harness](http-async-probe/run_probe.py) and exact [compile](http-async-probe/compile.json) and [link](http-async-probe/link.json) metadata are retained unchanged. [compile.log](http-async-probe/compile.log) and [link.log](http-async-probe/link.log) are intentionally empty: both commands exited 0 without diagnostic output. This probe wrapper was linked only into diagnostic ELF `3020146aa9df003d18f863c314f748d96bf9bf62e646b4fbc6f96ac84980c3af`, which remained unchanged during execution. It is not part of the packaged application. The link record verifies that the existing packaged executable, debug file and SD server archive stayed unchanged during the isolated relink; the packaged executable's SHA-256 was `12e68947a6bdb61aedc1c76f3075c84a50b5ba72c0e2941e0597df10010e0374`.

| Controlled check | Observed result |
| --- | --- |
| Progress while the first two-second task was active | 32 successful polls; maximum measured call duration 0.001740563999 seconds |
| Conflicting image generation and options mutation | HTTP 409 for each while work remained active |
| Synthetic completion through the real response handler | HTTP 200 with the expected synthetic string; task marked complete |
| Client disconnect after the second task became active | Task later completed and the server answered a subsequent ping |
| SIGTERM while the third task was active | Clean process exit 0 after 2.020151758 seconds; no forced cleanup kill |
| Entire probe | `status: passed`, 6.515541692 seconds |

All three started/finished wrapper markers are retained in stderr. The final pending HTTP connection closes during the deliberate shutdown test; the report separately verifies that the process exits cleanly with work active at signal time. It does not require that shutdown return a completed image response to that last client. The measured poll durations describe this small controlled run, not a general latency guarantee under actual model load.

This result addresses responsiveness, busy-request rejection, response lifetime after disconnect, and orderly shutdown around the actual asynchronous route/dispatcher. It does not exercise SD graph construction, the enlarged model-worker stack, sampling callbacks, cancellation of a real model, PNG encoding, WebGPU inference or Windows. The subsequent [real-model CPU API run](api-async/report.json) is separate and is now archived below. Its pass comes from actual model inference and early cancellation, not this synthetic replacement. Final Windows/CI inference gates remain pending.


## Completed real CPU HTTP image, progress and early cancellation

The [original API report](api-async/report.json), [server stdout](api-async/server.stdout.log), [server stderr](api-async/server.stderr.log), [PNG](api-async/api-image.png), and [BUILD.json](api-async/BUILD.json) preserve the **passing local real-model API run**. It used packaged application `12e68947a6bdb61aedc1c76f3075c84a50b5ba72c0e2941e0597df10010e0374`, the external pinned trained SD1.5 Q4_0 checkpoint, and the CPU backend. No synthetic model-work wrapper was linked into this executable. Total verifier time was **254.557 seconds**.

The first request used the apple prompt, 256 × 256, four steps, seed 42, CFG 7, Euler and discrete scheduling. It returned HTTP 200 with one real decoded PNG and the matching task ID. The PNG is **120,155 bytes**, SHA-256 `de9199ded8368659705badde3e529dd593396c5da87d50e6d362d4b8edd47c39`. Its chunk CRCs and nonconstant RGB pixels were independently verified, and its recorded `quality_assessed` remains false. Its identical bytes to the earlier partial API run are a local observation, not a promised cross-platform image hash.

| Real-model lifecycle check | Recorded result |
| --- | --- |
| Active progress responsiveness | 32 successful polls while the first request was active; maximum 0.002233 seconds against a 2-second per-request timeout |
| First image completion | Step 4 of 4; `completed=true`, `interrupted=false`; one PNG returned |
| Concurrent generation and options mutation | HTTP 409 while generation was active |
| Wrong/finished-task interruption and duplicate task ID | HTTP 409 |
| Invalid request overrides | Three informative HTTP 400 results, with gate release and persistent options unchanged |
| Separate cancellation request | Eight steps requested; sampling observed at completed step 1 before interruption |
| Task-specific interrupt | HTTP 200; the generation request then returned HTTP 200 with no images |
| Cancelled task state | Step 1 of 8, `interrupted=true`, `completed=true`; early cancellation observed and gate released |
| Actual inference worker stacks | Two measured 8,388,608-byte stacks, each with a 4,096-byte guard, queried through `pthread_getattr_np` |
| Process lifecycle | Server remained alive after verification, then exited 0 during harness cleanup |

The native engine prints error-level sampling messages when its computation is deliberately interrupted. In this run those lines immediately follow the matched interrupt and `unet graph execution cancelled`; the API records the expected cancelled task and returns no images. They are preserved rather than removed, and do not represent an unhandled generation failure in this passing cancellation test. The first image request completed before cancellation of the separate task.

The report retains the checkpoint override, the unchanged persistent options, the complete HTTP event sequence, and verified Generate-page script hashes. Native single-user capabilities and unsupported kiosk behavior were checked. `browser_execution_tested=false`: this verifies served resources and HTTP behavior, not a browser interaction. The active-poll timings are observations from this run, not a service latency guarantee. This API run does not measure application RSS or WebGPU execution.

Both application and model SHA-256 values were unchanged after execution. The copied build metadata identifies a local dirty build based on `c6b876179c88ad02c811d81b81fb08c79efc2cdf`; it must not be relabeled as a clean build of the subsequently published implementation commit `6898a4e564c3363d1692f0aed1a2da0395a17a77`. A new CI build and its native Windows/Linux reports are separate evidence. This local pass establishes the tested CPU API inference/lifecycle scope and does not assert final cross-OS validation.
