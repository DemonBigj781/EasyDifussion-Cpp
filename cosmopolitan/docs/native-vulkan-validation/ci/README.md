# Final native-provider CI evidence

[Run 37568302821](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37568302821) completed successfully in all four jobs. The retained [offline inspection](inspection.json) passed all **22 evidence groups** for the same application, built from clean source `1730424270e35ff4a06bd978654c9990cfd5777e`:

- Application: `easy-diffusion.exe`, **190,491,826 bytes**.
- Application SHA-256: `7dd513293f53bc39bd7c92e3aaff0522c89f0d543c3390b4dd95e2f590b507b9`.
- [BUILD.json](application/BUILD.json), [SYMBOLS.json](application/SYMBOLS.json) and [static dependency metadata](application/software-webgpu/LINK.json) bind the build, shared symbols and source/dependency pins.

## Navigation and tested scope

| Evidence | What it establishes |
| --- | --- |
| [Linux isolated runtime](linux/results-linux-isolated/report.json), [Windows native PE runtime](windows/results-windows/report.json) | Same-file application self-tests, direct GGML and trained llama execution, shared backend checks and existing runtime/API gates. Linux isolation and Windows PE execution are separate launch modes. |
| [Linux shell bootstrap](linux/results-linux-bootstrap/report.json) | Shell launch with transient extraction of the embedded ELF loader and unchanged application bytes; distinct from explicit-loader isolation. |
| [Linux CPU image](linux/results-inference-cpu/report.json), [Linux embedded WebGPU image](linux/results-inference-webgpu/report.json), [Windows CPU image](windows/results-inference-cpu/report.json), [Windows embedded WebGPU image](windows/results-inference-webgpu/report.json) | Real pinned-model CLI image completion, independently checked PNGs and applicable generation/denoising counters. Low-step images are plumbing checks, not quality benchmarks. |
| [Linux CPU inference API](linux/results-inference-api/report.json), [Windows CPU inference API](windows/results-inference-api/report.json) | Real image requests, responsive progress, ownership/busy gates and early cancellation. These API runs use CPU inference. Windows harness cleanup terminates the process; it does not establish graceful Windows shutdown. |
| [Linux configuration HTTP/restart](linux/results-configuration-api/report.json), [Windows configuration HTTP/restart](windows/results-configuration-api/report.json) | Actual configuration routes, persistence and restart behavior. [Linux configuration unit](diagnostics/config-test/report.json) and [Windows configuration unit](windows/config-report.json) each retain 38 production-module observations. |
| [Native Linux device](linux-native/results-native-device/report.json), [native/embedded coexistence](linux-native/results-dual-provider/report.json) | Actual direct tensor graphs and trained llama inference through host software Vulkan; both providers coexist in one graph-test process. Separate llama invocations are identified in their reports. |
| [Native Linux diffusion API](linux-native/results-native-inference-api/report.json) | Real pinned-model image completion through native software Vulkan after the loader-worker correction. Counters cover the whole inference closure, not a denoising-only interval. Cancellation was explicitly skipped in this native API run. |
| [Windows native DLL discovery](windows/results-windows-native-discovery/report.json) | Actual installed Chrome SwiftShader DLL loading and explicit rejection for missing ShaderF16. This is native discovery/rejection evidence, **not native GPU computation**. |
| [Native main/loader regression](diagnostics/native-main-executor-test/report.json) | Host TLS/native enumeration, 14 main-executor checks, 9 loader-policy checks and the legacy off-main fault reproduction. These focused checks are distinct from model/shader execution. |

The [build job log](build-job.log) also records the Generate DOM/request mock's `UI_DOM_MOCK PASS` and all eight Settings DOM/request mock cases. Those are JavaScript mocks; they do not claim browser, server or hardware execution.

**No physical GPU was available.** Native Linux used host Mesa/llvmpipe, while the portable fallback used embedded lavapipe. Hardware-required checks correctly reject software devices. Image/model execution may use CPU fallback; its fraction is not measured. Full operation/model coverage, image-quality equivalence and full training parity are outside this record.

The separate [development archive](../development/README.md) retains the failed `9ff6b99` application and diagnosis. The [local corrected CLI archive](../local/README.md) retains the exact previously failing command completed with this same `1730424` application. Its earlier statement that final HTTP CI evidence was pending describes the time that local archive was prepared; the completed native HTTP result is linked above.

## Provenance and preservation

[ARCHIVE.json](ARCHIVE.json) records the four job identities and five artifact IDs, ZIP sizes and SHA-256 digests. [Run](run.json), [job](jobs.json) and [artifact](artifacts.json) metadata, plus each download receipt, were checked by [archive_final.py](archive_final.py). The archive procedure rehashed the retained ZIPs, matched their digests/sizes to GitHub artifact metadata and compared every selected artifact member with the copied bytes before publication.

The frozen [FILES.json](FILES.json) inventories **190 files totaling 2,857,845 bytes**. Its SHA-256 is `401f03dcabdae1b80773175ae1bb63991c86ca2f267d21549124f7ba3746f0a7`. **This README and FILES.json itself are outside that frozen inventory.** The inventoried `.gitattributes` disables text conversion so archived byte hashes survive Windows Git checkouts.

This directory contains small metadata, verifiers, logs and output PNGs. It contains no application/loader, debug executable, native DLL, model checkpoint, static library or download ZIP. Recorded build-host archive/DLL/model hashes do not imply that those omitted binaries were freshly rehashed by a reader of this directory. Raw records are unchanged; logs lacking a runner-recorded hash were reparsed and inventoried during offline review, not retroactively hash-bound to their runner.

To reproduce the offline inspection, retain the **five original extracted artifacts** (application, diagnostics, Linux, Windows and Linux-native) and a checkout of source `1730424270e35ff4a06bd978654c9990cfd5777e`. Invoke [inspect_results.py](inspect_results.py) with their directories through `--app`, `--diagnostics`, `--linux`, `--windows` and `--linux-native`, plus:

```text
--expected-source 1730424270e35ff4a06bd978654c9990cfd5777e
--expected-sha256 7dd513293f53bc39bd7c92e3aaff0522c89f0d543c3390b4dd95e2f590b507b9
--parser-dir /path/to/matching/checkout/cosmopolitan/tests
```

The explicit parser directory is required for the archived layout. Inspection only parses saved evidence and checks bytes; it launches no application/model and uses no network. The original artifacts are necessary because the application and other verification binaries are intentionally absent here.
