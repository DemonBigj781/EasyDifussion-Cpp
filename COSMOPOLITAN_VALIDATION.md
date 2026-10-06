# Cosmopolitan native integration: verified artifact

**One unchanged x86-64 executable passed the native Windows and Linux runtime
checks on 2026-10-06.** The application contains one shared GGML, llama.cpp,
the custom diffusion engine, the partial native SD 1.5 trainer, the native
HTTP server and embedded C++ UI resources.

This is a completed CPU integration milestone within the
[whole-application port](cosmopolitan/PORTING.md). Full application API parity,
image generation from a diffusion checkpoint, end-to-end training,
LibTorch/ONNX tools, remaining Python replacements and GPU execution remain
separate completion gates.

## Exact artifact and provenance

| Item | Recorded value |
| --- | --- |
| Application | `easy-diffusion.exe` |
| Size | 109,465,883 bytes |
| SHA-256 of the executable | `f3d13f19691631d033ac45d442b197534f902a12017cecc0a1f2e1520dca9734` |
| Application source commit | `00e3cb3094db2ac548933268109d9b909e018171` |
| Host verifier commit | `10405079ea540116feb8079dccaed0c1fc2166fd` |
| SDK | Cosmocc 4.0.2, pinned download and checksum |
| Target | x86-64 Cosmopolitan APE; CPU baseline without an AVX requirement |
| Build producer | [Run 37511033658](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37511033658), successful build job `112431939190` |
| Application artifact | [easy-diffusion-cosmopolitan-x86_64, ID 11435639650](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37511033658/artifacts/11435639650) |
| Successful runtime verification | [Run 37514525238](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37514525238) |
| Windows job | [112443909853](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37514525238/job/112443909853) |
| Linux job | [112443910162](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37514525238/job/112443910162) |
| Machine-readable evidence | [COSMOPOLITAN_VALIDATION.json](COSMOPOLITAN_VALIDATION.json) |

The artifact ZIP also contains build metadata, the symbol-audit report, an
explicit Linux loader and the original host test scripts. The executable hash
above is not the ZIP hash. The successful recheck used the current verifier
from its recorded commit, checked the original clean build provenance, and
ran the original executable without rebuilding or modifying it. Both OS
reports record that same hash before and after execution.

The source changes after the application build and through the verifier
commit concern documentation, CI and the host verifier. They do not alter
the application implementation. A new build from another commit may have a
different hash because its embedded build provenance changes. The portability
claim concerns using the same produced file on both systems, not identical
output from independent builds.

GitHub Actions artifacts have the workflow's 30-day retention period. The
source, pins, build recipe and this evidence record remain in Git.

## What passed

| Area | Execution evidence | Boundary of the result |
| --- | --- | --- |
| Shared tensor implementation | One GGML source set; one definition of selected GGML, llama, diffusion and application entry symbols; consistent `GGML_MAX_NAME=160` | Does not establish a complete shared application scheduler or model cache |
| CPU numerical behavior | F32 and Q4_0 matrix multiplication, bias/RMS normalization/SiLU and softmax against scalar references at one and four threads | Covers the tested operations and shapes |
| GGML compatibility changes | 24 checks covering clamp, offset RoPE forward/inverse/gradient behavior, zero-dimension RoPE and SSM state history across one and four threads | Accelerator implementations of the added semantics remain separately gated |
| Language inference | Real model loading, tokenization, finite logits and 16 greedy decode steps matching the independent original llama/GGML reference | Uses the small trained `stories260K.gguf` regression fixture |
| Existing training mathematics | GELU, quick GELU and SiLU backward checks; separate AdamW rates; sampling leaves weights and subsequent updates unchanged | Does not run a complete LoRA training recipe or validate checkpoint quality |
| Diffusion integration | Linked diffusion API initialization and CPU backend/device checks | Does not generate an image from a full diffusion checkpoint |
| Native command behavior | Device listing, diffusion/trainer help and precise rejection of an unmatched trainer option | Does not establish every command option or model family |
| UI and server | All 16 C++ UI pages render with embedded assets; actual HTTP requests for the home page, training page, CSS and backend-device JSON pass | Many UI controls still need their application services ported |
| Resource integrity | Embedded fixture/assets match their manifest hashes; executable hash stays unchanged | User models, datasets, configuration and outputs remain external data |

The llama fixture is pinned to the `ggml-org/tiny-llamas` revision
`def3e2dd70df35ecbf6403ea347de4c5977220c1`, with SHA-256
`047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b`.
The independent reference generated these token IDs for `Once upon a time`:

```text
432,383,286,261,376,298,315,421,395,317,426,338,401,396,267,337
```

Both OS reports match that sequence exactly. Reference-build provenance is
recorded in [llama-reference.json](cosmopolitan/docs/llama-reference.json).

The HTTP checks received 200 responses for `/`, `/cpp-ui/training`,
`/cpp-ui/assets/ui.css` and `/v1/sdapi/v1/backend-devices`, and rejected the
encoded parent-traversal request. The native server, renderer and model
libraries are linked into the application; the full text/image/training HTTP
service and shared job lifecycle still require integration.

## How the two operating systems were tested

**Windows Server 2022:** native Python launched the PE directly from a fresh
application directory containing only `easy-diffusion.exe` and a temporary
directory. The child had a restricted environment and a System32-only PATH.
It did not use WSL, a Linux container, a second compiled application or a
deployed Python interpreter. Windows system facilities remain host dependencies.

**Ubuntu 24.04, isolated filesystem:** the host verifier created a real chroot
containing only the application, an explicit APE loader and a temporary
directory. The child replaced the host Python process with the loader and
application. No Python installation, source checkout or host shared libraries
were copied into that filesystem. This checks deployment dependencies; it is
not a security boundary for running untrusted programs.

**Linux single-file bootstrap:** a separate test started the application using
`/bin/sh` without supplying an external APE loader. Its restricted PATH
contained only the ordinary bootstrap utilities. The shell header extracted
the bundled `.ape-1.10` ELF loader transiently: 9,249 bytes, SHA-256
`cc31af13515fcf091768bfbf12da899286b35ff1bf1302f304936db4b03d512e`.
The application file remained unchanged. This test uses host shell utilities
and is distinct from the isolated-filesystem test.

The completed checks target x86-64 Windows and Linux. They do not establish
macOS, BSD, ARM, every Windows/Linux version or every processor configuration.

## Windows runtime conventions found during validation

The producer run's first Windows verifier failed even though the application
completed its self-tests. The failure led to two concrete fixes in the host
harness, followed by the successful unchanged-artifact run linked above.

1. **Inherited merged-file logging:** Cosmopolitan 4.0.2 can assign stdout and
   stderr separate cursors even when a native Windows parent redirects them
   to the same regular file. Their output can overwrite each other. The
   verifier now gives them distinct files, retains both raw streams and
   combines labeled diagnostic sections only after child cleanup. This is a
   documented runtime limitation; it has not been hidden by weakening marker
   checks or by changing the application to guess which handles are aliases.
2. **Exit-status representation:** the pinned runtime deliberately passes
   `application_exit_code << 8` to native Windows termination. The malformed
   trainer command therefore produces exactly `512` for native Windows
   Python and `2` for Linux Python. The verifier requires the exact
   platform-specific value and the expected diagnostic, and records the raw
   status alongside the expected application code. Arbitrary nonzero or crash
   statuses are not accepted as that expected failure.

The [run guide](cosmopolitan/README.md) includes the separate-output-file
workaround and links to the pinned runtime sources. Numerical, token,
return-code and integrity checks remain enforced.

## Earlier update attempts: what the source audit established

The [source audit](cosmopolitan/docs/source-audit.md) compared complete Git
trees and blob identities against the revisions recorded by the repository.
It found no missing `src/` or `include/` entry in the audited llama, diffusion
or diffusion-GGML snapshots.

The llama core matches its recorded upstream pin. Its three content changes
are build-metadata adjustments; its four omitted entries are development,
benchmark or Apple packaging files. `Theory` and `main` have the same llama
subtree, so copying that directory from `Theory` would not recover a newer
core. This finding does not mean the pin is the latest upstream version.

The diffusion and GGML snapshots contain intentional local changes, including
the custom training work. Those changes were retained as the source of the
shared runtime. The unfinished integration was real: the standard native
build isolated llama because it used a different GGML, and the application
still depended on Python orchestration and services.

The portable build reconciles the required llama interfaces into the custom
diffusion GGML. It also fixes an offset-RoPE backward-graph propagation defect,
separates an unrelated HTTP model-index type from the diffusion weight manager,
adapts the server's Asio use to the pinned runtime and links the native entry
points and UI resources. The original vendored snapshots remain preserved
inputs to deterministic staging. Details are in the
[shared-GGML guide](cosmopolitan/shared-ggml/README.md).

## Remaining work toward the whole application

- Generate an image from an actual diffusion checkpoint on both systems and
  validate the supported inference options and model families.
- Run the partial native SD 1.5 trainer end to end, including export and
  sampling, then port the additional training families and documented options.
- Connect text, image and training requests to one application job manager,
  with progress, cancellation, model lifetimes and a shared memory budget.
- Replace the remaining Python application services and required conversion,
  tokenizer, processor and Transformers behavior before removing those paths.
- Integrate or deliberately replace the LibTorch Sprite-GPT/vision and ONNX
  tagging dependencies, accounting for their allocators and tensor semantics.
- Connect the previously verified software Vulkan/WebGPU foundation and
  validate each new backend's actual inference and backward operations.
  Hardware driver support and multi-GPU placement require their own tests.

The API audit found 168 Python route declarations across 128 scanned Python
files, four mounts and 20 native literal routes. These counts include legacy
and plugin declarations and are not a count of unique active endpoints or a
feature-completion percentage. See the
[application API inventory](cosmopolitan/docs/application-api.md).

[Colibri and KoboldCpp](cosmopolitan/docs/resource-management.md) were reviewed
as references for model residency, resource ownership and application behavior.
Their implementations were not imported. Their current platform packages do
not supply this branch's same-file deployment or remove the need for a shared
native application scheduler. Specialized attention and training constraints
are tracked in [attention.md](cosmopolitan/docs/attention.md).

## Build and run

Build a new artifact on the supported x86-64 Linux host using the pinned recipe:

```sh
python3 cosmopolitan/build.py --jobs 2
```

For the verified downloaded artifact, Windows can run:

```powershell
.\easy-diffusion.exe --self-test
.\easy-diffusion.exe llama --prompt "Once upon a time" --tokens 32
.\easy-diffusion.exe sdkit --backend cpu --port 8188
```

Linux can run the same file using its bundled bootstrap:

```sh
/bin/sh ./easy-diffusion.exe --self-test
/bin/sh ./easy-diffusion.exe llama --prompt "Once upon a time" --tokens 32
/bin/sh ./easy-diffusion.exe sdkit --backend cpu --port 8188
```

The native UI is served at `http://127.0.0.1:8188/`. The default llama model
is the small regression fixture; pass `--model PATH` for an external supported
GGUF model. See the [run guide](cosmopolitan/README.md) for command options,
the logging caveat and the distinction between rendering pages and completing
their application services.
