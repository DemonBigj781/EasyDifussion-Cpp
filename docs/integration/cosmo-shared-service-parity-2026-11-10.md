# Restored Cosmo application staging and request parity

## Build blocker and semantic loss

A replay of the complete current Cosmo source/application patch series found
zero-fuzz failures in app patches 0004 and 0013. Changes in the merged native
server and task-state code no longer matched the old extraction contexts.
Rebasing deletion hunks alone would have hidden another problem: the compiled
Cosmo ApplicationServices copy did not contain newer main-branch behavior.

The patches now reconcile current main source while retaining Cosmo's shared
request admission, request-local settings overlays, worker dispatch, backend
selection, cancellation and service lifecycle. Shared services carry forward:

- Prompt validation before task admission and global error reset.
- ToMe and negative-guidance minimum-sigma controls and validation for both
  image and video requests.
- Native motion-module input and video latent/VAE interposer options, including
  existing Furception validation and encode/decode paths.
- NGMS/cache compatibility checks and related response metadata.
- Phase progress fields alongside existing step/preview progress.

Image ControlNet/Union, LLLite, IP-Adapter and interposer behavior remain intact.
The request/parsing helper implementations are reconciled, not replaced with
new model or backend abstractions. No video LoRA or model-generation test is
part of this change.

## Patched-header compile correction

Fully materialized patched headers exposed a separate non-Cosmo Asio fallback
bug: its forwarding macro expanded socket constants to numbers before token
pasting, producing names such as ASIO_OS_DEF_1. Source patch 0001 now pastes the
original argument directly in that fallback. The Cosmopolitan-specific branch
is unchanged. No original vendor source is edited.

## Regression approach

The new application-patch suite copies only tracked patch targets into a fresh
temporary tree, then applies all 13 patches in production order with fuzz zero.
It does not inspect model or large tokenizer data. Tests compare the extracted
request/progress handlers and parsing helpers exactly with their pre-extraction
versions, preserving Cosmo's earlier patch-layer changes. This prevents future
main controls from disappearing when the extraction patch deletes their source.
Additional checks enforce preflight ordering, shared transport delegation, and
compile/run real task-state completion, interruption and phase behavior.

Host compilation checks exercise actual shared-service and patched HTTP
transport sources with the patched application/Crow headers. These are compiler
checks, not a linked Cosmocc artifact or end-to-end model inference. The upstream
foundation SDK/export remains an independent prerequisite for full packaging.

## Remaining feature work

This restores current main/Cosmo parity for these handlers; it does not claim
that main itself implements every UI option. Confirmed follow-on gaps include
forwarding distilled_cfg_scale to native distilled guidance and honoring the
legacy UI's mask_blur request. Neither is silently enabled by this patch.

Validation on 2026-11-10: 127 foundation/application tests passed with no skips,
including seven new patch/handler/task-state/header-compilation tests. The full
13-patch replay uses fuzz zero and reports no rejects. Host compile checks use
fully patched Asio/Crow/app headers, not a fallback to original headers.
No model files, GPUs, Media drives or video LoRAs were used. Complete Cosmocc
linking/packaging and Windows execution still depend on the foundation work.
Both the real shared-service translation unit and patched HTTP transport also
compiled successfully to native host object files with the fully patched header
fixture. This is stronger than syntax-only verification but is not final linking.
