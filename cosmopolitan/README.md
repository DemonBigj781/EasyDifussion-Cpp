# Cosmopolitan Test: native Easy Diffusion integration

This branch builds a single x86-64 Actually Portable Executable containing
one shared GGML implementation, llama.cpp, the custom diffusion engine,
the partial native SD 1.5 trainer, the native HTTP server, and the C++ UI with
embedded resources. The shared GGML also includes its WebGPU backend, linked
to wgpu-native, Mesa lavapipe and LLVM inside that executable. New command
dispatch and llama orchestration use C; existing C++ and Rust implementations
remain linked in the same process.

**Status: experimental native integration.** Application API parity,
end-to-end image/training validation, LibTorch/ONNX tools, remaining Python
replacements and hardware acceleration have separate completion gates.
The new GGML WebGPU integration is implemented, but its application runtime
and same-artifact Windows/Linux validation are pending. Earlier CPU-only
application results and standalone software-Vulkan tests do not establish
that this new integration passes.
The full scope is in [PORTING.md](PORTING.md).

## Build

The supported build host for this recipe is x86-64 Linux. The GitHub workflow
uses Ubuntu 24.04. Its host prerequisites are a C/C++ toolchain, Git, Python 3
with venv support, curl, CA certificates, unzip, patch, binutils, file,
bison, flex, pkg-config, CMake, Ninja and Clang/libclang. The workflow records
the corresponding Ubuntu packages. Run from the repository root:

```sh
python3 cosmopolitan/build.py --jobs 2
```

The script verifies and installs Cosmocc 4.0.2, prepares the pinned source
trees and patches, installs the pinned host CMake/Ninja versions, builds the
complete static software WebGPU dependency set, compiles the application,
and embeds UI assets, notices and a small trained llama validation model.
Downloads require network access during the build. Python and the build tools
are host dependencies.

[software-webgpu/PIN.json](software-webgpu/PIN.json) fixes the foundation to
`cosmopolitan-lua` commit `c2a7b3f8c871e44ae1e3fc8db36759c7b1829326` and verifies
its source archive hash. That pin supplies wgpu-native 29.0.1.1, Mesa 25.2.8,
LLVM 19.1.7, Cosmocc 4.0.2 and Rust `nightly-2026-07-28`. The recipe installs
the Rust toolchain in its build directory, builds the patched Rust standard
library and compatibility objects, and source-builds the required LLVM/Mesa
archives. Mesa's host shader compiler and Python generators are build tools;
the embedded LLVM JIT executes shaders at runtime. There is no dependency on
an installed Mesa ICD, native Vulkan library or display server for this path.

The output is `cosmopolitan/out/easy-diffusion.exe`. `BUILD.json` records its
hash, component/patch provenance, static WebGPU dependency metadata and
embedded-resource hashes. Debug symbols
are retained separately in `easy-diffusion.com.dbg`. Build products, downloaded
tools and model weights are excluded from Git.

Large diffusion translation units are serialized to bound peak compiler
memory. The generated vocabulary data is compiled in 12 translation units,
preserving the original loader bodies and data bytes. An 8 GiB build host
requires that serialization; increasing the global job count does not
parallelize those large units.

The dependency build is substantial on its first run. Its default cache is
`cosmopolitan/out/software-webgpu`; later builds validate the pinned source
and artifact identities before reuse. `--jobs` also sets the Rust, LLVM and
Mesa job limits. Useful build switches:

```sh
python3 cosmopolitan/build.py --prepare-only
python3 cosmopolitan/build.py --dependencies-only --jobs 2
python3 cosmopolitan/build.py --libraries-only --jobs 2
```

`--prepare-only` stages sources without compiling the dependency set.
`--dependencies-only` completes that set without building the application.
`--libraries-only` still requires the WebGPU dependencies, then builds the
shared model/UI libraries. `--software-webgpu-out PATH` selects a dependency
cache; `--foundation-source PATH` supplies local Git objects for the exact
pinned commit. A missing or failed WebGPU dependency fails the build; the
recipe does not produce a CPU-only replacement artifact.

## Run the application

On Windows, invoke the executable directly:

```powershell
.\easy-diffusion.exe --self-test
.\easy-diffusion.exe llama --prompt "Once upon a time" --tokens 32
.\easy-diffusion.exe llama --backend webgpu --prompt "Once upon a time" --tokens 32
.\easy-diffusion.exe webgpu-test
.\easy-diffusion.exe sdkit --backend cpu --port 8188
```

On Linux, the portable shell entry point is:

```sh
/bin/sh ./easy-diffusion.exe --self-test
/bin/sh ./easy-diffusion.exe llama --prompt "Once upon a time" --tokens 32
/bin/sh ./easy-diffusion.exe llama --backend webgpu --prompt "Once upon a time" --tokens 32
/bin/sh ./easy-diffusion.exe webgpu-test
/bin/sh ./easy-diffusion.exe sdkit --backend cpu --port 8188
```

The shell entry point extracts the executable's bundled APE loader into a
temporary directory. The application file stays unchanged. It requires the
ordinary shell/bootstrap utilities; the separate isolated-filesystem test
uses an explicitly supplied APE loader and requires no shell or host shared
libraries inside the application root.

The native server's C++ UI is at `http://127.0.0.1:8188/`. Rendering a page
does not establish that every control's backend service has been ported;
see [application-api.md](docs/application-api.md).

Other commands:

```text
--help                       Show command dispatch
--version                    Show the integration build
ggml-test                    Run scalar-reference CPU tensor checks
webgpu-test                  Run direct WebGPU tensor and trained-model checks
llama --help                 Text generation options
sdkit --help                 Existing native server/tool options
sdkit --list-devices          List the linked backend devices
train --help                 Supported partial native training options
render-ui /training          Render one existing C++ UI page
```

`llama` defaults to the embedded small trained fixture. Use `--model PATH`
for an external supported GGUF model. The fixture is a regression input,
not an image checkpoint or a production assistant. User models, datasets,
configuration and generated outputs remain external data. A host browser
displays the UI.

`llama --backend cpu|webgpu` defaults to `cpu`. Selecting `webgpu` requests
all-layer offload to the linked WebGPU device and requires actual backend
dispatch/readback evidence; an unavailable WebGPU device is an error. The
server accepts `sdkit --backend webgpu --port 8188` and routes that selection
through the same shared registry. This routing alone does not establish
complete diffusion-model or training support.

The current WebGPU device is **software execution on the CPU** through
lavapipe and LLVM. It exercises the real WebGPU command/shader path but does
not provide physical GPU acceleration. `WebGPU0` is the GGML device name;
its description identifies the embedded software adapter. GGML's ordinary
CPU backend remains a separate choice.

### Windows redirected logging with SDK 4.0.2

If a native Windows launcher supplies stdout and stderr handles for the same
regular file, the pinned runtime can overwrite one stream with the other.
This occurred in the initial CI harness with Python's `stderr=STDOUT`:
the application returned success, but some earlier self-test output was lost.
It is a runtime logging limitation, not merely a verifier parsing issue.

For file capture, use separate destinations. For example, in `cmd.exe`:

```bat
easy-diffusion.exe --self-test >self-test.stdout.log 2>self-test.stderr.log
```

The host verifier now captures distinct files and combines labeled diagnostic
sections only after the child exits; those sections do not represent a merged
chronological stream. All numerical, token, exit-code and integrity checks
remain enforced. SDK 4.0.2 initializes the inherited descriptors with separate
[cursors](https://github.com/jart/cosmopolitan/blob/4.0.2/libc/intrin/fds.c)
and uses each cursor for explicit-offset
[Windows writes](https://github.com/jart/cosmopolitan/blob/4.0.2/libc/calls/readwrite-nt.c).
The application does not infer handle aliasing from matching file names or
inodes, which would conflate deliberately independent opens of the same file.

### Native Windows exit-status encoding

Cosmopolitan 4.0.2's [exit implementation](https://github.com/jart/cosmopolitan/blob/4.0.2/libc/intrin/exit.c)
stores the application's exit code shifted left by eight bits in the native
Windows process status. A native Windows parent therefore observes `512`
when the application returns `2` for invalid arguments; Linux reports `2`.
Zero still means success on both systems. The verifier checks the exact
expected native status and records the application code separately; it does
not accept an arbitrary nonzero result as the expected failure.

## One shared GGML

The authoritative runtime starts with the custom diffusion GGML tree and
backports the llama interfaces and CPU semantics it needs. The original
snapshots remain inputs to deterministic staging. Both engines and the
trainer compile against the same tensor layout (`GGML_MAX_NAME=160`),
operation definitions, allocator interfaces and backend registry.

The reconciliation includes clamp semantics, RoPE offsets and their backward
graph propagation, SSM state history, conservative backend capabilities and
static backend registration. See [shared-ggml/README.md](shared-ggml/README.md)
for pins, patches and numerical checks.

The recipe enables CPU and the existing shared GGML WebGPU implementation.
A limited C++ API adapter forwards its original WGSL kernels and resource
operations to the pinned wgpu-native C API. It does not introduce a second
GGML or replace shader dispatch with CPU tensor routines. See
[backend/README.md](backend/README.md) for the adapter and callback contract.
Hardware Vulkan, CUDA, Metal, SYCL and other accelerator implementations are
not enabled, and unrelated backend plugins are not loaded into the process.

The pinned WebGPU kernels require `ShaderF16`, including F32 matrix kernels
that stage inputs through half-precision workgroup memory. Device registration
fails when the software adapter lacks that capability. The pinned native C
implementation does not expose the standard subgroup capability/information,
so workgroup kernels are used and subgroup-dependent flash attention is
rejected. Dawn-only subgroup matrices, packed integer-dot kernels and optional
GPU timestamp profiling are disabled. Existing per-operation capability
checks still apply, including shared-GGML restrictions for custom RoPE offsets
and SSM history. An x86-64 CPU-backend baseline therefore does not promise that
every x86-64 machine can run this ShaderF16-dependent WebGPU path.

## Validation and its limits

The workflow builds the application once, records its SHA-256, and passes
that artifact to separate Windows and Linux runtime jobs. Windows executes
the PE directly. Linux checks both the shell bootstrap and operation in a
fresh filesystem containing only the application, the explicit APE loader
and a temporary directory. The host Python verifier is not copied into the
isolated application filesystem.

The separate `Cosmopolitan existing-artifact verification` workflow can run
the current host verifier against an earlier application artifact without
rebuilding it. Its dispatch inputs pin the build run, executable SHA-256 and
clean source commit; each platform checks that provenance before execution.
Use matching values for the artifact being checked. The expanded verifier
requires WebGPU execution, so the historical CPU-only artifact from build run
`37511033658` cannot satisfy the new backend contract.

The checks cover:

- One authoritative GGML source set and exactly one definition of selected
  tensor, model and application entry-point symbols.
- F32 and Q4_0 tensor computations against an independent scalar reference,
  at one and four threads.
- Clamp, offset RoPE forward/inverse/gradient behavior, and SSM state history.
- Real llama model loading, tokenization and 16 greedy decode steps, matched
  to an independently built original llama/GGML reference.
- Mandatory direct WebGPU F32 and Q4_0 matmul, bias, RMSNorm, SiLU and softmax
  graphs with two changed-input cases each. Twelve readbacks are compared with
  independent scalar references. These graphs use only WebGPU buffers and
  direct backend execution, with no CPU scheduler fallback; an unsupported
  backward operator must remain rejected.
- A separate trained llama prefill and 16-step decode with WebGPU selected,
  finite logits and the same independent greedy-token reference. Both tensor
  and model tests require real graph, submission, dispatch, matrix-dispatch
  and readback counter deltas, an embedded software adapter and zero native
  Vulkan loader opens. Adapter enumeration or success markers alone do not pass.
  A second model counter snapshot after synchronized prefill must also show
  positive graph, submission, dispatch, matrix-dispatch and readback deltas
  during the sixteen decode steps, so prefill alone cannot satisfy the test.
- The repository's existing native backward-math and text-encoder optimizer
  checks, plus diffusion API/device initialization.
- Rendering all 16 C++ UI pages from embedded assets; native server startup
  and HTTP requests; rejection of malformed trainer input.
- Embedded model/resource integrity and unchanged application hashes after
  execution.

Run the host checks after building:

```sh
python3 cosmopolitan/tests/audit_symbols.py --out cosmopolitan/out
python3 cosmopolitan/tests/verify_runtime.py --artifact-dir cosmopolitan/out
python3 cosmopolitan/tests/verify_bootstrap.py --artifact-dir cosmopolitan/out
sudo python3 cosmopolitan/tests/verify_runtime.py --artifact-dir cosmopolitan/out --isolate
```

`--self-test` includes both mandatory WebGPU tests as well as the existing
CPU/model/training/UI regressions. `webgpu-test` runs only the new tensor and
model pair. The tensor fixture uses inputs exactly representable in f16 to
isolate graph correctness from the preserved matrix kernel's input rounding;
it does not claim arbitrary-F32 equivalence to CPU matmul.

These checks do not load a full diffusion checkpoint, complete a training
recipe, validate every model architecture, establish all UI/API behavior or
exercise a physical GPU. The trainer still defaults to CPU; WebGPU coverage
of custom backward, F8 and optimizer operations is not established. Its
existing no-fallback behavior and documented partial-training limits remain
applicable. Actual execution reports, rather than a workflow name or a
successful compile alone, determine which gates passed.

## Audit and design references

- [Earlier source updates](docs/source-audit.md): actual Git object comparison
  for llama.cpp, diffusion and custom GGML, plus unfinished integration.
- [Application services](docs/application-api.md): route inventory, UI
  behavior, queue/resource ownership and Python replacement.
- [Colibri and KoboldCpp](docs/resource-management.md): model residency,
  streaming, packaging and relevant licensing boundaries.
- [Attention](docs/attention.md): current custom attention, training
  requirements and Sol-Attn applicability.
- [Independent llama reference](docs/llama-reference.json): exact fixture,
  prompt, token IDs and original native-build provenance.
- [Fixture licensing and purpose](MODEL.md).
