# Native Vulkan development evidence

This directory preserves the first native-software Vulkan validation results and the diagnosed diffusion loader failure. It is development evidence, separate from validation of the corrected application. The original files are retained byte for byte; their source paths, sizes and SHA-256 hashes are recorded in [FILES.json](FILES.json). The inventory contains **37 files, 436,049 bytes**. This README, the inventory itself and the archive's newline-preservation attributes are outside that inventory.

## First CI application and scope

[Workflow run 37565714314](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37565714314), native Linux job `112617658775`, tested this application:

| Field | Recorded identity |
| --- | --- |
| Source commit | `9ff6b99e2f1f754b441b38126bcfdebbed704c31` |
| Source state | Clean (`dirty=false`) |
| Application | `easy-diffusion.exe`, 190,487,399 bytes |
| Application SHA-256 | `b5f1ef5e13adf049e8e938d8ab157807c7748fff9a2934080c5d44ae5e5acf6e` |
| Downloaded application ZIP | 95,782,686 bytes |
| ZIP SHA-256 | `399ca78689ac081d0b560d942d8682577d84115b5b798c4418c5009d3df2c6db` |
| Native adapter | llvmpipe, Mesa `25.2.8-0ubuntu0.24.04.4`, LLVM `20.1.2` |
| Host loader package | `libvulkan1:amd64 1.3.275.0-1build1` |

[BUILD.json](first-ci/application/BUILD.json), [SYMBOLS.json](first-ci/application/SYMBOLS.json), the [application download verification](first-ci/application-download-verification.json), [diagnostics download verification](first-ci/diagnostics-download-verification.json) and [driver package record](first-ci/linux-native/native-driver-packages.txt) bind the reports to their original build and host driver. The application, debug executable, host driver, model weights and dependency archives are intentionally not copied here. Their recorded hashes do not imply that this evidence archive contains or freshly audits those binary artifacts.

**All adapters in these results are software CPU implementations.** Native Vulkan means calls through the host Vulkan loader/driver; it does not mean physical GPU acceleration. Every recorded adapter has `software=true`, WebGPU adapter type `3`, and `hardware_verified=false`. Hardware-required negative checks returned application status `1` and were recorded as successful rejections, not hardware passes.

The remaining Windows and isolated Linux jobs from this superseded run were cancelled after the correction was published. This directory makes no claim that their image gates passed. Corrected-source validation belongs to the later run built from `1730424270e35ff4a06bd978654c9990cfd5777e` and must be assessed from that run's own reports.

## What passed before the failure

The [native-device report](first-ci/linux-native/results-native-device/report.json) passed four real direct GGML F32/Q4_0 graphs and twelve independent scalar comparisons on `native/WebGPU0`, with no CPU scheduler fallback. Its actual totals were 4 graphs, 16 submissions, 20 dispatches, 4 matrix dispatches and 12 readbacks; the native loader-open count was 1. The graph command took 5.927 seconds.

The regular llama CLI on the same device completed the trained embedded fixture's sixteen greedy tokens, matching the independent reference:

```text
432,383,286,261,376,298,315,421,395,317,426,338,401,396,267,337
```

The llama command took 3.020 seconds. Whole-inference counters were 17 graphs, 51 submissions, 1,785 dispatches, 782 matrix dispatches and 17 readbacks. The separately measured sixteen decode steps had 16 graphs, 48 submissions, 1,680 dispatches, 736 matrix dispatches and 16 readbacks. Normal llama scheduling can include CPU operations; these are actual WebGPU execution counters, not an assertion of universal offload. The raw [graph stdout](first-ci/linux-native/results-native-device/graph.stdout.log) and [llama stderr](first-ci/linux-native/results-native-device/llama-0.stderr.log) retain the scalar errors, token IDs and scoped execution records.

The [dual-provider report](first-ci/linux-native/results-dual-provider/report.json) also passed. Under `provider=auto`, one graph-command process created both providers and ran all four graphs on each: `embedded/WebGPU0` and `native/WebGPU1`. Each device produced the same graph-count tuple **4 / 16 / 20 / 4 / 12**, for eight graph cases and twenty-four scalar comparisons overall. The command took 1.517 seconds. Its [raw stdout](first-ci/linux-native/results-dual-provider/graph.stdout.log) includes per-device boundaries and successful completion of both adapters. A native loader-open count of 1 in this mixed process is expected, including while measuring embedded execution; this is not the separate zero-native-open embedded isolation gate.

The harness then launched separate regular llama commands for both selectors. Both matched the sixteen reference tokens. Embedded llama whole/decode counter tuples were **17 / 51 / 1,785 / 782 / 17** and **16 / 48 / 1,680 / 736 / 16**; native llama whole/decode tuples were **51 / 119 / 1,802 / 782 / 51** and **48 / 112 / 1,696 / 736 / 48**. Those two llama invocations are not claimed to run in the same process as one another. The report records exact durations and commands.

Both successful graph/llama reports record the unchanged application SHA-256 before and after.

## Real diffusion API failure and direct reproduction

The [native diffusion API report](first-ci/linux-native/results-native-inference-api/report.json) **failed**. Startup, device listing and initial request ownership checks succeeded, but the first full-model image request lost its connection when the application exited with native status `-11` (SIGSEGV). Total harness elapsed time was 5.814 seconds. The report records unchanged application and model hashes. No image was returned and no diffusion/API success is claimed.

The [server stdout](first-ci/linux-native/results-native-inference-api/server.stdout.log) records allocation of 1,667.59 MiB of model parameter buffers and then stops at `2/196` text-encoder tensors. That log calls non-host backend buffers “VRAM”; here they belong to a software Vulkan driver, so it is not evidence of physical GPU memory. The [server stderr](first-ci/linux-native/results-native-inference-api/server.stderr.log) records an 8 MiB HTTP service thread and a 16 MiB original-main inference stack. Merely moving the outer generation call to main did not move the loader's internally created workers.

A local [direct image-command reproduction](local-model-crash/direct-report.json) used the **same CI application hash** and pinned full SD1.5 model, with native `WebGPU0`, 256×256 output, two requested steps and two loader threads. It exited `-11` after 18.540 seconds. The raw [stdout](local-model-crash/direct.stdout.log), [crash stderr](local-model-crash/direct.stderr.log), [symbolization metadata](local-model-crash/symbolization.json) and [symbolized call stack](local-model-crash/symbolization.txt) preserve the diagnosis:

```text
std::__thread_proxy<ModelLoader::load_tensors ...>
  ModelLoader::load_tensors worker
  ggml_backend_webgpu_buffer_set_tensor
  wgpuQueueWriteBuffer
  wgpu_core queue write / staging-buffer allocation
  wgpu_hal Vulkan create_buffer
```

The crash report identifies main process/thread ID 4 and faulting worker thread ID 18. Symbolization confirms the actual model-loader worker reached the native buffer path. The top shared-library frame remains unresolved; the evidence does not assign an invented native function name to that address. This is an off-main host-libc TLS failure diagnosis, supported by the focused legacy-worker reproduction below, not the earlier unrelated HTTP stack-overflow diagnosis.

A separate [post-reproduction hash capture](local-model-crash/post-reproduction-hashes.json), recorded at `2026-10-07T03:49:36.615435+00:00`, independently rehashed the application and full model after the direct crash. Both still matched the identities recorded above. This later capture is retained separately; the original direct report is unchanged.

The model SHA-256 in both full-model attempts is `c2f6e92f9d08d69cc673a1003528ac8199274b3c0eaec88d5fbefe5af67bd42b`, for the 1,747,190,784-byte pinned SD1.5 Q4_0 checkpoint. The blocked strace attempt, its empty trace file and generated native helper files are omitted; the retained direct report explicitly records `traced=false`.

## Focused correction evidence

The correction executes the generic SD loader worker synchronously on original main for Linux native/auto policy. CPU-only conversion/export workers explicitly remain host-only, while embedded-only and Windows loading retain their existing worker behavior. The published patch and helper are [0012-native-model-loader-main-thread.patch](../../../../cosmopolitan/app-patches/0012-native-model-loader-main-thread.patch) and [native_main_executor.cpp](../../../../cosmopolitan/src/native_main_executor.cpp). Those links identify implementation files; the byte-preserved reports below remain the evidence for this development run.

The [focused regression report](loader-regression/report.json) passed **14 existing main-executor checks and 9 model-loader checks**. It exercised actual host TLS and Vulkan loader enumeration, inline native-policy work, unchanged embedded policy, wrong-thread rejection before invoking work, exception transfer and shutdown. The [stdout](loader-regression/stdout.log) records both PASS markers; [stderr](loader-regression/stderr.log) retains the observed execution details.

The probe executable's recorded SHA-256 was unchanged: `e6a2500419f16bdd02ee037f8c4d4246b72546b9347ec8dfc6f61a7805ebaaae`. Its host TLS fixture SHA-256 was `4721d1dadcc1d4226050397d2fc68ab533ba6605230ff9014fbe000803a5979c`. Neither binary is included here. The separate legacy-loader-worker negative case reproduced `-11`; its [stderr](loader-regression/legacy-worker.stderr.log) identifies the worker before its host TLS call.

**This focused regression is not a full checkpoint-upload, diffusion-image or shader pass.** It tests the real main-thread policy/helper and the specific host TLS hazard. Successful full native diffusion inference, corrected same-file Windows/Linux gates and any physical GPU coverage require their own later evidence.
