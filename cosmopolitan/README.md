# Cosmopolitan Test: native Easy Diffusion integration

This branch builds a single x86-64 Actually Portable Executable containing
one shared GGML implementation, llama.cpp, the custom diffusion engine,
the partial native SD 1.5 trainer, the native HTTP server, and the C++ UI with
embedded resources. New command dispatch and llama orchestration use C;
existing C++ implementations remain linked in the same process.

**Status: experimental native integration.** Application API parity,
end-to-end image/training validation, LibTorch/ONNX tools, remaining Python
replacements and accelerator integration have separate completion gates.
The full scope is in [PORTING.md](PORTING.md).

## Build

The supported build host for this recipe is x86-64 Linux. The GitHub workflow
uses Ubuntu 24.04. Install Git, Python 3 with venv support, unzip, patch and
binutils, then run from the repository root:

```sh
python3 cosmopolitan/build.py --jobs 2
```

The script verifies and installs Cosmocc 4.0.2, prepares the pinned source
trees and patches, installs the pinned host CMake/Ninja versions, compiles
the static application, and embeds UI assets, notices and a small trained
llama validation model. Downloads require network access during the build.
Python and the build tools are host dependencies.

The output is `cosmopolitan/out/easy-diffusion.exe`. `BUILD.json` records its
hash, component/patch provenance and embedded-resource hashes. Debug symbols
are retained separately in `easy-diffusion.com.dbg`. Build products, downloaded
tools and model weights are excluded from Git.

Large diffusion translation units are serialized to bound peak compiler
memory. The generated vocabulary data is compiled in 12 translation units,
preserving the original loader bodies and data bytes. An 8 GiB build host
requires that serialization; increasing the global job count does not
parallelize those large units.

Useful build switches:

```sh
python3 cosmopolitan/build.py --prepare-only
python3 cosmopolitan/build.py --libraries-only --jobs 2
```

## Run the application

On Windows, invoke the executable directly:

```powershell
.\easy-diffusion.exe --self-test
.\easy-diffusion.exe llama --prompt "Once upon a time" --tokens 32
.\easy-diffusion.exe sdkit --backend cpu --port 8188
```

On Linux, the portable shell entry point is:

```sh
/bin/sh ./easy-diffusion.exe --self-test
/bin/sh ./easy-diffusion.exe llama --prompt "Once upon a time" --tokens 32
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

The current recipe enables the CPU backend. It does not yet connect the
previously verified software-Vulkan/WebGPU foundation, hardware Vulkan,
CUDA, Metal, SYCL or the other accelerator implementations. Backend plugins
compiled against unrelated GGML versions are not loaded into this process.
Each added accelerator needs its own capability and numerical validation,
including backward operations needed by training.

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
The initial defaults identify build run `37511033658`. Use matching values
from a later build when rechecking a different executable.

The checks cover:

- One authoritative GGML source set and exactly one definition of selected
  tensor, model and application entry-point symbols.
- F32 and Q4_0 tensor computations against an independent scalar reference,
  at one and four threads.
- Clamp, offset RoPE forward/inverse/gradient behavior, and SSM state history.
- Real llama model loading, tokenization and 16 greedy decode steps, matched
  to an independently built original llama/GGML reference.
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

These checks do not load a full diffusion checkpoint, complete a training
recipe, validate every model architecture, establish all UI/API behavior or
exercise a GPU. The partial trainer's supported options and limitations
remain applicable. Actual execution reports, rather than a workflow name or
a successful compile alone, determine which gates passed.

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
