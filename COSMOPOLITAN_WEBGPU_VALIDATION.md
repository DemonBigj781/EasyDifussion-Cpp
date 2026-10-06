# Cosmopolitan WebGPU integration: verified Windows/Linux artifact

**One unchanged x86-64 executable passed native Windows and isolated/bootstrap
Linux checks on 2026-10-06, including actual shared-GGML WebGPU tensor graphs
and trained-model prefill and decoding.** The executable contains one shared
GGML, llama.cpp, the custom diffusion engine, partial trainer, native HTTP
server, embedded UI, wgpu-native, Mesa lavapipe and LLVM.

This completes the recorded software-backend integration milestone. WebGPU
executes on the CPU through embedded lavapipe/LLVM; physical GPU acceleration,
full diffusion generation, end-to-end training and whole-application service
parity remain separate gates. The historical
[CPU-only validation record](COSMOPOLITAN_VALIDATION.md) remains unchanged.

## Exact CI artifact and provenance

| Item | Recorded value |
| --- | --- |
| Application | `easy-diffusion.exe` |
| Size | 190,143,210 bytes |
| Executable SHA-256 | `7d74b7f67750f6c6397ed3da95b9c3803ee030d08c975791c651acefc912481a` |
| Source and host verifier commit | `8d22e47b692d8865d703b7227a6288cc078817b6` |
| Packaged source state | Clean: `dirty=false` |
| SDK | Cosmocc 4.0.2, pinned archive and checksum |
| Target | x86-64 Cosmopolitan APE |
| Successful build and runtime run | [37531113555](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37531113555) |
| Build job | [112500630596](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37531113555/job/112500630596) |
| Windows job | [112525334275](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37531113555/job/112525334275) |
| Linux job | [112525334307](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37531113555/job/112525334307) |
| Application artifact | [11448230211](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37531113555/artifacts/11448230211) |
| Completed workflow last updated | 2026-10-06 22:07:08 UTC ([run metadata](cosmopolitan/docs/webgpu-validation/run.json)) |
| Machine-readable evidence | [COSMOPOLITAN_WEBGPU_VALIDATION.json](COSMOPOLITAN_WEBGPU_VALIDATION.json) |

The downloaded executable's size and SHA-256 match the clean source provenance
in [BUILD.json](cosmopolitan/docs/webgpu-validation/application/BUILD.json). Windows, isolated Linux and
Linux bootstrap each report that same executable hash before and after use.
The bootstrap copy also retains that hash. There was no per-OS application
rebuild. The portability result concerns one produced file, not byte-identical
output from independent builds.

The application artifact ZIP is 94,422,877 bytes, with SHA-256
`5d1d23c0b547930761c39cb51b294d5f9225b906328bf1d1d16072422af90ee1`.
That is the archive hash, not the executable hash. Results are retained in
Windows artifact `11447296662` (18,008 bytes; archive SHA-256
`1cbc273b459092f5182d09235c69fe5ee52bbe9875931cad3124fb19bcb9413f`) and
Linux artifact `11447966062` (24,584 bytes; archive SHA-256
`22f4517eff07a00270ce3702bf157d44559ca4c210505b9bf54aba04e4eca518`).
GitHub artifact expiry is recorded in the original
[artifact listing](cosmopolitan/docs/webgpu-validation/artifacts.json). Small reports, metadata and logs are
also retained in this repository, byte-for-byte, with the
[file hash inventory](cosmopolitan/docs/webgpu-validation/FILES.json). No application or loader binaries are
stored in that evidence directory.

## The three tested launch modes

| Mode | Actual report | Self-test elapsed time | Deployment evidence |
| --- | --- | ---: | --- |
| Windows native PE | [Windows report](cosmopolitan/docs/webgpu-validation/windows/report.json) | 6.531 s | Windows Server 2022; initial directory contained only the executable and `tmp` |
| Linux isolated filesystem | [Isolated report](cosmopolitan/docs/webgpu-validation/linux/results-linux-isolated/report.json) | 2.876 s | Real chroot with only the executable, explicit APE loader and `tmp`; `isolated=true` |
| Linux shell bootstrap | [Bootstrap report](cosmopolitan/docs/webgpu-validation/linux/results-linux-bootstrap/report.json) | 2.773 s | `/bin/sh` entry, fresh TMPDIR, limited utility PATH, no supplied external loader |

Windows launched the PE directly with a System32-only PATH, without WSL or a
Linux subprocess. The Linux isolated test's child process entered the chroot
and executed the APE loader and application. No Python installation, host
shared libraries or source checkout was copied into that root.
The displayed platform strings describe the verifier's host environment, not
a dynamic glibc dependency of the application.

The separate shell test allowed only `uname`, `mkdir`, `dd`, `gzip`, `chmod`
and `mv`. It extracted the bundled `.ape-1.10` x86-64 ELF loader transiently:
9,249 bytes, SHA-256
`cc31af13515fcf091768bfbf12da899286b35ff1bf1302f304936db4b03d512e`.
The original and copied executable stayed unchanged; the temporary directory
was removed. This mode uses host shell utilities and is distinct from chroot.
Elapsed times describe these runs and are not a performance benchmark.

## Actual tensor dispatch and numerical results

The direct probe initialized the shared GGML WebGPU backend and allocated its
non-host buffers. It used direct backend graph computation with no CPU backend
or scheduler fallback. Each F32/Q4_0 case performed matmul → bias → RMSNorm →
SiLU → softmax with K=64, N=11, M=13. Iteration 1 changed the input values.
All twelve scalar-reference checks returned 143 finite values each.

The following printed maximum absolute errors were independently read from
each CI raw log and are identical across Windows, isolated Linux and bootstrap:

| Input type | Iteration | Matmul error | Bias/RMSNorm/SiLU error | Softmax error |
| --- | --- | --- | --- | --- |
| F32 | 0 | 0 | 1.84800599e-07 | 5.11464063e-08 |
| Q4_0 | 0 | 0 | 1.87900205e-07 | 5.79208283e-08 |
| F32 | 1 | 0 | 2.06550482e-07 | 5.16552808e-08 |
| Q4_0 | 1 | 0 | 2.81014707e-07 | 5.09649803e-08 |

Each of the four cases recorded exactly **1 graph, 4 submissions, 5 dispatches,
1 matrix dispatch and 3 mapped readbacks**, with `cpu_fallback=0`. The
unsupported `SILU_BACK` operation remained unadvertised. Every case identified
the embedded software provider and zero native Vulkan loader opens.

The per-element acceptance bound was `3e-5 * (1 + abs(reference))` for matmul
and `4e-5 * (1 + abs(reference))` for the other stages. The table gives actual
observed errors. Direct-test inputs and Q4 reconstruction are exactly
representable in f16, because the preserved matrix kernel stages even F32
inputs through half-precision workgroup memory. This establishes the tested
graphs, not arbitrary-F32 equivalence to CPU matmul. The Q4 oracle uses the
original F32 right-hand matrix, not CPU quantized-dot Q8 rounding.

Raw evidence: [Windows stdout](cosmopolitan/docs/webgpu-validation/windows/self-test.stdout.log),
[isolated Linux stdout](cosmopolitan/docs/webgpu-validation/linux/results-linux-isolated/self-test.stdout.log),
and [bootstrap log](cosmopolitan/docs/webgpu-validation/linux/results-linux-bootstrap/self-test.log).

## Trained-model prefill and sixteen decode steps

The embedded trained `stories260K.gguf` fixture was loaded, `Once upon a time`
was tokenized into five prompt tokens, and sixteen decode steps completed
with finite logits for a 512-token vocabulary. CPU-selected and WebGPU-selected
inference matched the independent original llama/GGML reference on both OSes:

```text
432,383,286,261,376,298,315,421,395,317,426,338,401,396,267,337
```

The fixture is pinned to `ggml-org/tiny-llamas` revision
`def3e2dd70df35ecbf6403ea347de4c5977220c1`, SHA-256
`047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b`.
It is a small trained regression model, not an image checkpoint or production
assistant. The independent reference-build provenance remains in
[llama-reference.json](cosmopolitan/docs/llama-reference.json).

The actual model counters agree in all three CI reports:

| Counter scope | Graphs | Submissions | Dispatches | Matrix dispatches | Readbacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill and sixteen decode steps | 17 | 51 | 1,768 | 782 | 17 |
| Sixteen decode steps after synchronized prefill | 16 | 48 | 1,664 | 736 | 16 |

The second snapshot proves dispatch/readback during decoding itself; prefill
activity alone cannot satisfy it. Both scopes recorded the embedded software
provider and zero native Vulkan loader opens.

Ordinary llama scheduling still includes CPU work and buffers. In the actual
[Windows stderr](cosmopolitan/docs/webgpu-validation/windows/self-test.stderr.log) and
[Linux stderr](cosmopolitan/docs/webgpu-validation/linux/results-linux-isolated/self-test.stderr.log),
WebGPU-selected inference reports 6/6 layers offloaded, a 0.99 MiB WebGPU model
buffer, a 0.12 MiB CPU model buffer, WebGPU and CPU compute buffers, and two
graph splits. The result proves WebGPU matrix execution during prefill and
decode, not universal operator offload. Only the separate direct tensor probe
has the explicit no-CPU-scheduler-fallback guarantee.

## Static implementation, source identity and capability limits

Both OSes identify adapter `llvmpipe (LLVM 19.1.7, 256 bits)` and driver
`Mesa 25.2.8 (git-c2ecdc20ef) (LLVM 19.1.7)`. This is CPU software Vulkan
execution of the real WebGPU command/shader path. GGML uses the scheduling
name `WebGPU0`; its description explicitly identifies the software adapter.
The [static dependency manifest](cosmopolitan/docs/webgpu-validation/application/software-webgpu/LINK.json)
records no runtime shared-library dependencies or external Vulkan loader for
this path. The original GGML WGSL kernels are retained, with a limited C++
adapter forwarding operations to the pinned wgpu-native C API.

| Component | Pin |
| --- | --- |
| Foundation | `cosmopolitan-lua` commit `c2a7b3f8c871e44ae1e3fc8db36759c7b1829326` |
| Foundation archive SHA-256 | `e7a9f22763e6d7b5777dd200ad00ab31a6d9c8afa877fec0b145ff8bb8d730be` |
| wgpu-native | 29.0.1.1, `6aed50955d934ac36049ba8d002034841633ae02` |
| WebGPU C headers | `673658bc2bd70ec39fc55ebe6bb0173cf6d0a603` |
| Mesa / LLVM | 25.2.8 / 19.1.7, source hashes and port patches in the retained manifests |
| Rust | `nightly-2026-07-28`, patched standard library/runtime |
| llama | `c589f0ed10c643678c4707dd160c21ac7633ebc0` |
| Authoritative GGML base | `e20c3a14aa70ee84ca58499814206dd08d8026bc`, with the recorded local/shared-runtime patches |

C implements the application entry/llama orchestration, direct tensor tests,
parts of GGML/Mesa and compatibility glue. C++ supplies existing model engines,
server, UI, trainer, GGML WebGPU code, adapter and LLVM. Rust supplies wgpu-native
and its dependencies/runtime; WGSL supplies compute kernels. These components
are linked in one process. This is not an all-C implementation. The recorded
closure includes 61 LLVM archives and 77 Rust compatibility objects. Python,
CMake, Meson, host compiler tools and glslang are build dependencies; embedded
LLVM performs shader JIT work at runtime without an external compiler process.

The [symbol audit](cosmopolitan/docs/webgpu-validation/application/SYMBOLS.json) passed one authoritative
`ggml-webgpu.cpp`, consistent `GGML_MAX_NAME=160`, no alternate versioned GGML
symbols, and one definition of each of the twenty selected GGML/model/app/
wgpu/lavapipe/LLVM symbols. This is source/symbol identity evidence; numerical
correctness is established by the separate runtime checks above.

The original WebGPU matrix kernels require `ShaderF16`, even for F32 tensors;
the device is not advertised when that capability is absent. The CPU backend's
baseline without an AVX requirement does not promise this WebGPU path on every
x86-64 processor. The pinned native C API does not expose standard subgroup
mapping/information, so subgroup-dependent flash attention is unavailable.
Dawn subgroup matrices, packed integer-dot kernels and optional timestamp
profiling remain disabled. Custom offset RoPE and SSM history capabilities
remain CPU-only. Existing operator/type/shape checks remain authoritative.

## Existing application regressions retained

Both full OS runtime reports passed CPU F32/Q4_0 graphs at one/four threads,
shared clamp/RoPE/SSM compatibility checks, CPU llama reference inference,
activation-backward and separate-AdamW/sampling tests, linked diffusion API
initialization, shared-registry WebGPU selector resolution, and all sixteen
embedded UI page renderings. Device listing and command help also passed.
No full diffusion checkpoint or training recipe was loaded by these checks.

The native server regression used the CPU selection and returned HTTP 200 for
`/`, `/cpp-ui/training`, `/cpp-ui/assets/ui.css` and
`/v1/sdapi/v1/backend-devices`; encoded parent traversal returned 404.
The malformed trainer command emitted the expected missing-`--threads`-value
diagnostic. Its raw status was exactly **512 on Windows / application code 2**
and **2 on Linux / application code 2**, matching the pinned SDK convention.
Separate stdout/stderr files avoid the known Windows merged-file cursor issue;
combined logs are labeled diagnostics, not a chronological merged stream.

## Separate earlier local artifact and public CLI check

The local build preceded CI and has a distinct identity:
source `240b564614831601ab69d159abc837cd3a288f56` (clean), 190,138,894 bytes,
SHA-256 `6f4ac2e658581315ee14ccaa139fef35fa76cf822422e397630c42bc16e5305b`.
Its focused WebGPU test, full local explicit-loader checks and shell bootstrap
passed. The local explicit-loader report was `isolated=false`; it was not the
chroot proof now provided by the CI artifact. Local reports are summarized
separately in the machine-readable record.

Only this local artifact was separately exercised through the public command:

```sh
cosmopolitan/out/ape-x86_64.elf cosmopolitan/out/easy-diffusion.exe \
  llama --backend webgpu --prompt "Once upon a time" --tokens 16
```

It returned exit 0 in 2.271 seconds and printed:

```text
Once upon a time, there was a little girl named Lily. She loved to play
```

The [local CLI report](cosmopolitan/docs/webgpu-validation/local/llama-cli.json),
[stdout](cosmopolitan/docs/webgpu-validation/local/llama-cli.stdout.log) and
[stderr](cosmopolitan/docs/webgpu-validation/local/llama-cli.stderr.log) preserve its exact evidence. Its
full/decode counters match the corresponding values above, and its single
recorded executable digest matches the local artifact. The report has no
separate before/after fields. This verifies the local public parser/backend
selection route; it is not claimed as a separately executed CI command. The
CI `--self-test` instead invokes the mandatory tensor/model test pair.

## Remaining work toward the whole application

- Physical GPU acceleration/native drivers and broader CPU/device/OS coverage.
- Full diffusion-checkpoint generation and supported architecture/options checks.
- End-to-end native SD1.5 training/export/sampling and additional training
  families. Native resume, AdamW8bit and selectable mixed precision remain
  incomplete; the partial trainer defaults to CPU.
- WebGPU support for custom backward, optimizer and F8 operations. Current
  inference evidence does not establish those capabilities, and unsupported
  operations/no-fallback training checks must remain enforced.
- Complete UI/API parity, shared job scheduling, cancellation, model lifetimes
  and memory ownership, remaining Python/Transformers services and LibTorch/ONNX.

These results establish the recorded x86-64 Windows Server 2022 and Linux
software-backend milestone. They do not establish every Windows/Linux release,
macOS/BSD/ARM support, every model, or full application completion.
