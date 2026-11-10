# Native request ownership and cancellation

## Production changes

Main now acquires an application admission lease before task creation, global
error reset, model selection/loading and image/video execution. A competing
request returns HTTP 409 instead of overwriting task state or changing model
selection between load and execution. Options writes, checkpoint/companion
refreshes and image-filter routes acquire the same gate. Progress reads and
interrupts do not acquire this execution gate.

Cancellation holds a separate identity lock through forwarding. A supplied
`id_task` must identify the active generation. Stale IDs receive 409 and cannot
cancel a successor admitted after completion. Empty, malformed or nonstring IDs
receive 400. For compatibility, Main still accepts a bodyless or empty-object
interrupt as “cancel the current request”; an idle legacy interrupt returns 200.
Main also retains sequential `default_task`/`default_video_task` reuse. Clients
needing isolation must use unique explicit IDs: identical reused IDs cannot
identify different generations, and the legacy bodyless endpoint is inherently
not task-isolated.

Preparing or completed tasks are not advertised as cancellable. The generator
now returns whether it actually accepted an interrupt; when its context is gone
(for example during fallback preparation), the task is not falsely marked
interrupted. Acceptance means a request was forwarded, not proof that a model
stopped. Main completion preserves interrupted sampling progress instead of
fabricating 100 percent, using the existing Cosmo behavior.

Cosmo retains its existing unique-ID, strict-interrupt and dispatcher policy,
without nesting a second Main admission gate. Its previously ungated refresh
and filter routes now use its existing generation mutex and dispatcher check.
Source staging explicitly replaces Main's policy with Cosmo's policy; all 13
patches continue to apply with zero fuzz. Both transports use the generator's
truthful interrupt result. This does not merge their intentionally different
legacy endpoint policies.

The Python native adapter owns each generated UUID across the HTTP call using
a lock-protected registry. Its production context is `threading.local`, so an
attribute on that context alone would not reach progress/stop threads. Progress
callbacks retain their original UUID, including delayed callbacks after a newer
request starts. Completion clears only its matching owner. Native idle stop
requests do nothing; external WebUI backends retain bodyless interrupts.
Native checkpoint and companion refresh calls run sequentially in one background
worker so sibling refreshes do not race against the admission gate.

## Verification

- Full CPU-only `sdkit` build and all 11 native CTests passed.
- The actual admission class is tested with threads and deterministic barriers:
  competing operations, matching/stale/legacy cancellation, failed-request
  unwinding, sequential default IDs, and completion blocked while an interrupt
  callback holds identity. The test passed 100 consecutive runs.
- The admission/task-state test passed ASan and UBSan. LeakSanitizer is disabled
  because it is unsupported under the executor's ptrace environment.
- Five real CPU HTTP test methods passed: existing 28 malformed generation
  cases, four bad task IDs, idle/stale/malformed interrupts, and two admitted
  parameter-parse failures proving completion, identity release and sequential
  reuse without reaching a model lookup.
- All 128 foundation tests passed, including full patch replay, host compilation
  and assertions that mutating routes require admission before resource access.
- All 46 Python identity, refresh, prompt, image, SAM and inpainting regressions
  passed with mocked transport where appropriate.

No model, GPU, video LoRA, full Cosmocc package or Windows execution was tested.
No sampling-time cancellation or full model inference is claimed.

## Remaining project-wide gates

The gate protects this server's HTTP resource users, not arbitrary direct calls
into ImageGenerator, multiple server instances sharing process-global callbacks,
or other out-of-band model users. Preparing cancellation is not sticky; callers
may retry after sampling begins. A refresh rejected because unrelated work is
active is still not automatically retried. Model-backed cancellation/fallback
and both live UI flows still need real model acceptance testing.

Theory's plan generator and RestAPI migration skeleton are not executable
inference replacements. Its optional GGUF resource owner must not replace the
working application's ModelManager/GGMLRunner ownership without further design.
The portable UI still reports missing img2img, companion-model and generation-
plugin UI capabilities. Python-backed training, download/conversion, gallery
and plugin parity, the complete foundation export, and full portable/Windows
package validation remain separate substantive work.
