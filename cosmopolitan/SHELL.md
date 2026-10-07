# One application, with a portable C shell

The application has a shared service layer for configuration, model discovery,
image inference, text inference, task progress and cancellation. The HTTP server
and the native shell call those services in the same process. The Generate page
uses that HTTP interface. They use one model index, one settings file and one
inference admission gate.

The new shell is implemented in C. Its command parser and file commands run
inside the application; they do not launch Node, Python, `/bin/sh`, `cmd.exe`,
or another copy of Easy Diffusion. [SHX](https://github.com/shelljs/shx) and
[ShellJS](https://github.com/shelljs/shelljs) are references for consistent
command spelling across operating systems. This implementation supports the
commands below and is not a drop-in implementation of either Node package.

## Start it

On Windows, start the application directly:

```powershell
.\easy-diffusion.exe shell --config .\easy-diffusion.json --serve
```

On Linux, the supplied APE loader starts the same application bytes:

```sh
./ape-x86_64.elf ./easy-diffusion.exe shell --config ./easy-diffusion.json --serve
```

The usual `/bin/sh ./easy-diffusion.exe ...` bootstrap also works. That initial
APE bootstrap is separate from the internal C shell. The native Vulkan loader
prerequisites described in [CONFIGURATION.md](CONFIGURATION.md) still apply.

`--serve` makes the UI and HTTP API available alongside the shell, at
`http://127.0.0.1:8188/` with the default settings. Omit it for a shell without
an HTTP listener. Both modes initialize the same application services.
`exit` or stdin EOF closes admission, finishes owned work and stops the HTTP
transport before returning. A shell session owns its application lifecycle;
starting and stopping multiple listeners within that session is not a command.

Use `--config PATH` when launching from different directories. The application
loads configuration once at startup. Shell commands do not repeatedly parse
the process arguments or reset the backend registry.

## Specify a model and a recipe once

Place a supported complete diffusion checkpoint in the configured checkpoint
directory. The default is `models/checkpoints` beside the configuration file.
Inside the application shell, the following commands have the same syntax on
Windows and Linux:

```text
models
defaults show
defaults save '{"options":{"sd_model_checkpoint":"stable-diffusion-v1-5-Q4_0.gguf"},"inference":{"image":{"width":256,"height":256,"steps":4,"seed":42,"sampler_name":"euler_a","scheduler":"discrete"}}}'
mkdir -p outputs
infer image --prompt "a red apple on a wooden table" --output outputs/apple.png
```

The checkpoint name must exactly match an indexed model. Replace the example
with a name returned by `models`. The application validates the checkpoint and
recipe, then saves them in one configuration-file replacement. A failed
validation does not save either half. The Generate page's **Save generation
defaults** button performs the same operation through
`POST /v1/sdapi/v1/settings`.

Later image requests inherit the saved checkpoint and recipe. An explicit
`--prompt`, `--steps`, `--model` or other request field applies only to that
request. It does not silently become a new saved default. To persist an
individual recipe field, use, for example:

```text
config set inference.image.steps 20
config set inference.image.prompt "a red apple on a wooden table"
options set sd_model_checkpoint "another-indexed-model.gguf"
```

`infer image` writes one PNG to the required `--output` path. The directory must
already exist, and the output file must be new. Its JSON response contains the
absolute `output` path, `output_bytes` and the native generation metadata. A
cancelled request that returns no image creates no output file.

Supported recipe defaults are prompt, negative prompt, width, height, steps,
guidance, seed, sampler and scheduler. Saved dimensions are multiples of 64
from 64 through 2048; steps are 1–200; guidance is 0–30; seed is -1 through
2147483647. A seed of -1 requests a fresh seed. Saved samplers are `euler`,
`euler_a` and `dpm++2m`, with `discrete` scheduling. This saved recipe is the
portable Generate form's supported subset; the existing native engine retains
its separately documented advanced request fields.

## Select compute explicitly when needed

Inspect the available devices and current configuration:

```text
devices
config show
```

CPU is the initial backend. To save automatic WebGPU selection for the next
application start:

```text
config patch '{"compute":{"backend":"webgpu","provider":"auto","device":"auto"}}'
```

The response reports `restart_required`; it preserves the current running
provider until restart. A request can choose an available device under the
already initialized provider with `--device SELECTOR`. An unknown selection
fails. Software adapters remain labeled as software CPU devices. A compatible
installed driver and the existing platform prerequisites are necessary for
native GPU use. These commands do not enable CUDA or establish that a physical
GPU was tested.

When specifying a device on a top-level invocation, provide its backend too:
`easy-diffusion.exe infer image --backend webgpu --device WebGPU0 --output image.png`.
Those startup overrides are validated before the registry is initialized.
The same backend/device pair is accepted inside the shell; it selects a device
from the provider that the running application already initialized.

## Text inference uses the same application

```text
infer text --prompt "Once upon a time" --tokens 16 --threads 1
infer text --model models/example.gguf --prompt "Hello" --tokens 64
```

The default model is the embedded `stories260K.gguf` regression fixture. It is
small trained test data, not a production assistant. An external model must
be supported by the linked llama engine. Results contain generated `text`,
`token_ids`, token counts, `task_id` and cancellation state. `--output FILE.txt`
also saves the generated text to a new file.

Text and image requests share admission: an active request receives the one
inference slot, and another request gets a busy error. Both engines use the
same GGML registry. The text service creates its model/context for each
request, releases request allocations, and leaves the application's global
backend lifetime intact. It does not yet maintain a resident language-model
cache or implement a multi-user conversation store.

## Application commands

| Command | Behavior |
| --- | --- |
| `status` | Application instance ID, active task ID and busy state. |
| `devices` | Actual shared backend/device metadata. |
| `models` / `models refresh` | Read or refresh the shared checkpoint index. Refresh is rejected during inference. |
| `config show` | Config path, saved values, effective values and restart requirement. |
| `config set DOTTED.KEY VALUE` | Save one supported field. An `options.*` key uses live option validation. |
| `config patch JSON` | Save a config patch; startup changes require restart, recipe changes are live. |
| `options show` / `options set KEY VALUE` / `options patch JSON` | Read/save live native generation options. |
| `defaults show` | Show the effective image recipe. |
| `defaults save JSON` | Atomically save an `{options, inference}` settings patch. |
| `infer image ...` | Shared image inference and PNG output. |
| `infer text ...` | Shared text inference and structured output. |
| `progress TASK_ID` | Read progress for the identified task. |
| `cancel TASK_ID` | Request cancellation of that task. |

Values for `set` are parsed as JSON when possible, otherwise as strings. For a
string that looks like a JSON number or boolean, preserve JSON quotes inside
the shell argument: `config set inference.image.prompt '"true"'`.

`infer image|text --request JSON` and `--request-file PATH` accept a native JSON
request instead of individual fields. A request object/file is limited to
1 MiB and duplicate keys are rejected by the command translator. Image
output still requires `--output PATH`. Use `help` for the individual options.

Inference is synchronous from the shell's perspective: its command returns
when that job finishes. HTTP/UI progress and task-specific cancellation remain
available while the shell waits. Conversely, a job submitted over HTTP is
visible to `status`/`progress` and can be cancelled from the shell. Diffusion
loading is still not cancellable; the text service has cooperative checks
during loading, prefill and decoding. Cancellation does not abort an already
running device dispatch halfway through.

## Portable file commands and syntax

| Commands | Scope |
| --- | --- |
| `pwd`, `cd PATH` | Inspect/change this shell's directory. `cd` never changes the process directory used by HTTP/model services. |
| `ls [PATH...]` | Sorted directory entries or specified files. |
| `mkdir [-p] PATH...`, `rmdir DIR...` | Create directories or remove empty directories. |
| `touch FILE...`, `cat FILE...` | Create/update file timestamps or print file bytes. |
| `cp SOURCE DEST`, `mv SOURCE DEST` | Copy a regular file or rename a file/directory. An existing directory destination receives the source basename. |
| `rm [-f] FILE...` | Remove files/symlinks; recursive directory removal is not implemented. |
| `test -e|-f|-d PATH` | Test existence, a regular file or a directory. |
| `echo [TEXT...]`, `true`, `false` | Print text or produce a command status. |
| `source FILE`, `exit [STATUS]` | Execute a script or close the session. |

Single and double quotes, `#` comments, newlines, `;`, `&&` and `||` are parsed
inside the application. Scripts stop at an unsuccessful command chain; `||`
can handle an expected failure. A terminal keeps accepting input after an
error. For example:

```text
test -d outputs || mkdir outputs
test -f outputs/apple.png && echo "Image exists"
```

Arguments are literal. Variable/glob expansion, pipes, redirection, background
operators and external process execution are not implemented. Use application
`--output` options for inference files. Prefer forward slashes in portable
scripts; native Windows absolute drive paths are accepted on Windows.

Use `shell -c SCRIPT`, `shell --file PATH`, or feed newline-delimited commands
to stdin. Script files support LF and CRLF. Limits are 1 MiB per script/input
line, 256 arguments per command, 4096 commands per parsed script and eight
nested `source` calls. NUL-containing input and invalid syntax are rejected
before executing that input. The probe checks shell behavior independently
of model/driver execution; full application checks have their own evidence.

## Implementation and verification boundaries

`application.h` is the C service boundary. `application_commands.cpp` translates
shell/CLI fields into requests; `ApplicationServices` implements the common
operations and owns the existing native model/task objects. `Server` adapts
HTTP requests to those operations. `shell.c` owns only parsing, file commands
and a shell-local directory. The Linux native Vulkan main-thread executor
continues to surround the shared service lifecycle.

The existing `image`, `llama` and numerical self-test commands retain their
diagnostic contracts. Product commands `infer image`/`infer text`, the portable
shell and native HTTP routes use the shared application service path. The
native trainer, conversion tools, remaining UI services, LibTorch and ONNX
components still have the completion gates recorded in
[PORTING.md](PORTING.md). This change does not claim those remaining subsystems
already share inference's full job lifecycle.

Focused verification entry points are `tests/verify_shell.py` and
`tests/verify_application.py`. The first tests the real C shell in a small APE.
The second tests one running application across shell and HTTP, repeated
trained text requests, settings persistence and optional real-model shared
image/cancellation work. The full integration workflow runs the same
application artifact on native Windows and Linux.
