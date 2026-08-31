# 30 — Live Runtime Argument Injection and Reload

## Objective
Allow supported sdkit/backend runtime options to be changed from Settings without manually editing launch commands and restarting the entire Easy Diffusion stack unnecessarily.

## Implementation
1. Classify every runtime argument as: live-mutable, requires model reload, requires backend process restart, or requires full application restart.
2. Replace raw free-form shell argument injection with a typed settings schema for known options. Keep an expert raw field only if necessary and validate it strictly.
3. Add a backend control endpoint/message for live-mutable values such as some logging or scheduling controls.
4. For model-reload settings, drain/cancel active jobs safely, unload model, apply settings, and reload.
5. For process-restart settings, supervisor should spawn a replacement sdkit with validated arguments, health-check it, then switch over/terminate old process.
6. Persist desired configuration separately from currently-active configuration and show pending restart state.
7. Reject incompatible combinations before disrupting a working backend.
8. Record effective arguments in diagnostics.

## Security
Never allow settings to become arbitrary shell command execution. Pass argument arrays directly to subprocess APIs.

## Validation
Each classification path, active-job behavior, invalid option rollback, backend failed restart, persistence, and UI state synchronization.

## Complete when
Supported runtime settings can be changed through Settings with the minimum required reload level, clear active/pending state, and safe rollback.
