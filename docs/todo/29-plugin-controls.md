# 29 — Per-Plugin Enable / Disable Controls

## Objective
Give users deterministic control over which plugins load, without deleting files or editing source.

## Implementation
1. Require stable plugin IDs in the unified plugin registry; generate a migration ID only for legacy plugins lacking one.
2. Store enabled state in persistent settings keyed by plugin ID and optionally version/source.
3. Apply disable state before executing plugin entry code so disabled plugins cannot register tabs/routes/events first.
4. Add UI listing plugin name, version, source, status, compatibility, conflicts, and last load error.
5. Support enable/disable with a clear indication whether restart/reload is required.
6. For plugins that can unload safely, add lifecycle hooks (`start`, `stop`, cleanup). Otherwise defer state change to next UI/server reload.
7. Protect required/core plugins from disabling unless a safe recovery path exists.
8. Add reset/safe mode that starts with optional plugins disabled for troubleshooting.

## Dependencies
Central plugin registry and deduplication.

## Validation
Disable before load, re-enable, conflicting plugins, crashing plugin, missing plugin after saved state, required plugin protection, safe mode, and settings migration.

## Complete when
Every optional plugin can be disabled predictably from the UI/settings and disabled code does not execute on startup.
