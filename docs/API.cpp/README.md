# API.cpp backend documentation

This tree documents GPU/backend-specific behavior that belongs behind the API.cpp abstraction layer. Application logic that is not backend-specific should not be documented here.

## Layout

Each backend uses the following structure:

- `backend/<backend>/attention/<attention_type>.md`
- `backend/<backend>/cache/easycache.md`
- `backend/<backend>/cache/teacache.md`

## Status vocabulary

Use these labels consistently:

- **Native** — implemented directly for the backend and intended for normal use.
- **Compatibility** — exposed through the common API but implemented through a compatibility adapter or translated path.
- **Fallback** — the API accepts the feature but routes to another implementation when the requested implementation is unavailable.
- **Stub** — symbols/interfaces exist, but useful backend execution is not implemented.
- **Unsupported** — intentionally unavailable for the backend or hardware target.
- **Not implemented** — no working implementation exists yet.

## Required fields

Every attention and cache page should record:

1. Current status.
2. Native implementation or adapter used.
3. Supported hardware/architecture targets.
4. Precision/data-type restrictions.
5. Compile-time requirements.
6. Runtime capability checks.
7. Fallback behavior.
8. Known limitations.
9. Validation/CI status.
10. Relevant source paths.

## Attention inventory

The common optimized-attention API currently identifies FlashAttention, SageAttention, and xFormers. A fourth requested attention implementation is still to be identified and added to the shared API; its documentation pages should be created once its exact implementation name is confirmed from the code/design work.

## Cache inventory

Each backend must document both EasyCache and TeaCache, even when the current status is unsupported or not implemented. This prevents missing backend work from being mistaken for completed integration.
