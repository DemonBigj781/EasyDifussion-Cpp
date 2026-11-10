# Cosmo build parity and source-foundation dependency

## Corrected application integration gap

The native SDKit build registered the app-owned prompt-validation runtime
adapter, but Cosmo adds stable-diffusion directly and bypasses SDKit's top-level
CMake. Its source glob included the JSON validator without adding the runtime
adapter, leaving sdkit_validate_prompt_attention unresolved at final link.

Both builds now call the same small runtime-adapters.cmake registration list.
The source stays in the existing stable-diffusion target; no second parser,
GGML runtime or model owner is introduced. Cosmo's staged SDKit source tree
contains this registration module with the rest of its source snapshot.

Three new regression tests check both build call sites and compile/link the
actual adapter source into static and shared host libraries. Their parser is an
explicit fixture, not inference. Existing native tests separately exercise the
real application parser. This does not claim a complete Cosmocc artifact build.

## Verified upstream dependency

On 2026-11-10 the remote foundation branch
[DemonBigj781/cosmopolitan dev/cosmopolitan-foundation-completion](https://github.com/DemonBigj781/cosmopolitan/tree/dev/cosmopolitan-foundation-completion)
remained at cb54f8f8a7fed3f9badb10f6749c9bfd9d63b6b6. That commit restores 18
architecture-specific binutils aliases, including x86_64-unknown-cosmo-ar needed
by the application consumer. cosmopolitan-lua is the repository's previous name;
its redirect and the separate master branch do not identify a newly accepted
foundation export.

The producer's [export documentation](https://github.com/DemonBigj781/cosmopolitan/blob/cb54f8f8a7fed3f9badb10f6749c9bfd9d63b6b6/toolchain/FOUNDATION_EXPORT.md)
and toolchain/export_foundation.py define the HANDOFF.json schema, relocatable
link metadata, materialized aliases, inventory and patch receipts. That source
revision documents fixture qualification, not a completed real SDK export or
application acceptance. No new complete export has been verified here.

The application therefore retains its existing immutable c2a7b3f8 foundation
pin. Promotion requires an actual source SDK and the application-compatible
Rust/WGPU/Mesa/LLVM dependency closure, including this application's WGPU patch
receipt. A compiler-only export or different branch head is insufficient.
The required application WGPU patch SHA-256 is
c0e164bdcf06faa800a6a500e8ae3e7c0ff3f41cf0be207771afc0e76a459c17.
Validate the complete fresh handoff with the existing consumer before a separate
pin-promotion change. The generated candidate is not a drop-in replacement for
the shared legacy PIN.json: its paths:["."] includes source symlinks that the
stock preparation route rejects, and default builds still select stock Cosmocc.
Promotion must explicitly select source-only routing or separate stock/source
pins. No upstream SDK/dependency build was started by this work.

## Validation and remaining parity

- Foundation/consumer suite: 120 host tests passed, including the three new
  registration/static/shared-link tests.
- Full native CPU application built; all eight native contracts passed Release
  and ASan/UBSan. LeakSanitizer remains disabled under executor ptrace.
- Live image API prompt rejection regression passed with no models loaded.
- The same application bytes running under Windows and Linux, the complete
  source-foundation export, and pretrained inference remain unverified here.
- Theory's unfinished tokenizer/conditioning/model-resource migration remains
  selective work; these build checks do not turn its prototypes into backends.

No GPU, Media-drive inspection or video-LoRA testing was performed.
