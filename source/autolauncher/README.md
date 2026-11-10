# Cosmopolitan autolauncher

Experimental, standalone bootstrap for a **staged** application bundle. It
selects a compatible executable set and supervises its processes. It does not
install dependencies, rewrite application settings, or replace `start.sh`.
Theory's application and GPU/model workflows have not been launched for this
work. APE/Linux fixture tests do not establish Windows/macOS release readiness.

## Build

From the repository root, using the previously documented Cosmopolitan 4.0.2
toolchain (substitute its actual extraction directory):

```sh
cmake -S source/autolauncher -B source/artifacts/autolauncher \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchains/cosmopolitan.cmake" \
  -DEDCPP_COSMOCC_ROOT=/absolute/path/to/cosmocc-4.0.2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build source/artifacts/autolauncher --parallel 2
```

Output: `source/artifacts/autolauncher/bin/easy-diffusion.com`. This is the APE
launcher; the `.dbg` and `.aarch64.elf` companions are not its release substitutes.
Python, GPU libraries and GGML are not linked into this executable.

An optional **native** CUDA-driver probe must be built separately:

```sh
cmake -S source/autolauncher -B source/artifacts/autolauncher-native \
  -DEDCPP_BUILD_CUDA_PROBE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build source/artifacts/autolauncher-native --target cuda-probe --parallel 2
```

It uses the installed driver dynamically, without CUDA SDK headers or nvcc.
Build one probe for each supported OS/architecture, with a conservative CPU
baseline. Never compile this native probe into the APE launcher. Windows builds
of the launcher itself use Cosmopolitan, not the native MSVC toolchain.

## Stage and package

The [example specification](package-spec.example.json) illustrates separate
locked uv CPU/CUDA application environments. **Its paths are placeholders for
release artifacts**, not a claim those environments or application entrypoints
exist in Theory. Stage the real workers, libraries, interpreter/environment
assets and pinned uv binary first, and adapt the spec to their real commands.
The existing Python application already owns SDKIT3 startup: do not also add
that same server as a launcher-managed service. Any app-specific backend
selection contract must agree with the selected bundle before release use.

```sh
uv run --no-project python scripts/package_autolauncher.py \
  --root /absolute/path/to/staged-bundle \
  --spec /absolute/path/to/package-spec.json \
  --launcher source/artifacts/autolauncher/bin/easy-diffusion.com
```

The packaging script only hashes staged files and copies the launcher. It does
not download or run any worker. It emits `launcher.json` next to the launcher;
use `--replace` to intentionally repack existing generated outputs. A normal
build Python also works; this script has no third-party Python dependencies.
Use pinned Python versions and tested lockfiles for the actual uv projects.
Offline examples require their interpreter, environments or installation cache
to be provisioned first; a lockfile alone does not provide those files.

For an explicit post-compile packaging target, configure the launcher build
with both `EDCPP_AUTOLAUNCHER_BUNDLE_ROOT` and
`EDCPP_AUTOLAUNCHER_BUNDLE_SPEC`, then build `autolauncher-package`. This target
deliberately repacks even when the launcher did not relink, so worker hashes
are refreshed. Publish completed bundles atomically with the application
stopped; the packager is not an in-place live updater.

## Invocation and selection

```sh
./easy-diffusion.com --detect
./easy-diffusion.com --list
./easy-diffusion.com --dry-run
./easy-diffusion.com
./easy-diffusion.com --backend cpu -- --application-option value
```

On Unix installations requiring APE's shell fallback, use
`sh -c './easy-diffusion.com ...'`. No loader registration, Wine changes, or
WSL-interoperability changes are performed. `--root PATH` overrides the default
bundle directory, which is the launcher's directory, **not** the shell's cwd.
`--manifest NAME` chooses a different manifest relative to that root.

- `--detect` reports runtime OS, executing ABI and CPU features without a manifest.
- `--list` checks compatibility, files and hashes without executing any child.
  Its `compatible` field means *static* compatibility; `probe_required` remains
  unevaluated and `probes_run` is false.
- `--dry-run` **executes bounded compatibility probes**, but no application or
  service. It prints the selected commands and candidate rejection reasons.
- Normal launch chooses highest priority first (manifest order breaks ties).
  `--bundle ID` and `--backend NAME` restrict selection but never bypass checks.
  Probe failure/timeout permits another compatible bundle, including CPU only
  if the packager explicitly supplied a CPU-capable set. No fallback happens
  after application/service startup has begun.
- Arguments after `--` are appended literally to the main program. There is no
  shell interpolation, concatenated command line, PATH search or automatic
  fallback to an arbitrary system Python.

## Manifest contract (schema 1)

The specification has `schema: 1` and a `bundles` array. Each bundle is a whole
compatible set; files from different bundles are never mixed.

| Field | Contract |
| --- | --- |
| `id` | Unique identifier. |
| `os`, `arch` | OS: `linux`, `win`, `mac`, `freebsd`, `openbsd`, `netbsd`, or `any`; ABI: `x64`, `arm64`, or `any`. Use `any` only for actually portable payloads. |
| `backend`, `priority` | Explicit backend name and integer priority (default 0). All non-`cpu` backends require a probe. |
| `cpu_features` | Optional requirements: `sse4.2`, `avx`, `avx2`, `fma`, `bmi2`, `neon`. AVX checks include OS register-state support. |
| `assets` | Packaging-only list of required scripts, lockfiles, libraries, etc. Executables are added automatically. Enumerate dependencies, not model caches. |
| `files` | Generated manifest map of relative paths to SHA-256; required for every executable/probe and listed asset. |
| `env`, `unset_env` | Optional environment additions/removals inherited by all commands in this bundle. |
| `probe` | Command object; exit 0 qualifies this bundle. `timeout_ms` defaults to 5000, maximum 30000. Use a baseline-safe native probe; do not run a GPU/ISA-specific worker merely to see if it crashes. |
| `programs` | Ordered command objects: zero or more `service` roles, then exactly one final `main`. |

Every command has an `id`, relative `executable`, optional string-array `args`,
relative `cwd` (default `.`), and optional `env`/`unset_env` overrides. Program
commands also have `role`. A service needs `ready_port` (IPv4 loopback TCP),
with `ready_timeout_ms` defaulting to 10000 (maximum 120000). An already occupied
port is refused. Readiness checks prove TCP acceptance, not application/API
health or endpoint identity; use dedicated local ports and add protocol-specific
validation before promoting a service requiring stronger readiness guarantees.

Only the explicit `{root}` token in arguments/environment values is expanded.
On Windows it becomes a native drive path. All other characters remain literal.
The launcher exports `EDCPP_LAUNCH_BUNDLE` and `EDCPP_BUNDLE_ROOT`. It clears
inherited `PYTHONHOME`/`PYTHONPATH` and sets `PYTHONNOUSERSITE=1`; a manifest can
explicitly override these. Other environment variables are inherited. Profiles
can run a pinned bundled `uv` with `run --locked`/`--frozen`, or a frozen Python
controller, just like any other program. The launcher does not remove Python.

For CUDA, use `cuda-probe --sm 86 --min-driver 12040 [--device 0]`. Repeat `--sm`
to list accepted exact compute capabilities. The driver value is CUDA driver
API compatibility (12040 means 12.4), not the NVIDIA driver package version.
The probe checks the chosen **visible device ordinal**, default 0. Set matching
device visibility/selection for the workers; passing a check on one GPU does
not validate unrelated GPUs. It does not allocate model weights or establish
that every required CUDA runtime library is present; list runtime dependencies
in assets and perform release integration tests. Other accelerator probes can
be supplied through the same bounded command contract.

## Process and trust boundaries

Child stdout/stderr are inherited for normal logs. Probe stdout is redirected
to stderr to keep inspection JSON clean, and probe stdin is closed. The main
exit status is propagated. SIGINT/SIGTERM/SIGHUP trigger cleanup; service
failure aborts startup/running main. Unix process groups get a grace period,
then SIGKILL. Workers must not daemonize into a new session to escape their
group. Windows joins a private, non-breakaway kill-on-close Job Object before
spawning; the non-inherited job handle is held until launcher exit to contain
descendants. Native Windows workers do not necessarily handle POSIX signals,
so graceful shutdown there requires worker-specific cooperation. Windows job
behavior is compiled but not runtime-validated in this Linux session.

Manifest paths cannot be absolute, contain traversal/backslashes/drive syntax,
or resolve through symlinks outside the bundle. Files must match recorded
hashes before a candidate runs. The manifest and installed bundle are trusted
local configuration: hashes detect corruption, **not** an attacker replacing
both manifest and workers. This is not a sandbox, signed update system, or
protection against concurrent file replacement. Do not store secrets in command
arguments that an explicit `--dry-run` prints.

Windows ARM remains deferred rather than blacklisted. Selection uses the
executing ABI, allowing a compatible x64 emulation path without claiming native
Windows ARM support. macOS, Windows and native ARM execution still need their
own validation; a multi-architecture build alone does not provide it.

## Tests

```sh
python3 source/autolauncher/tests/test_launcher.py \
  --launcher source/artifacts/autolauncher/bin/easy-diffusion.com --ape
```

The suite stages disposable Python fixtures and exercises process management;
it never imports Easy Diffusion, loads models or starts real GPU workers. It
also runs against a native Unix launcher build without `--ape`. Linux x86-64
is the runtime test host. Keep tests separate from application validation.

Dependencies are reused without editing vendored sources: nlohmann JSON
(`stable-diffusion.cpp/thirdparty/json.hpp`, MIT) and the public-domain SHA-256
implementation under `llama.cpp/vendor/hash`. Retain their attribution when
distributing the launcher, along with Cosmopolitan's notices.
