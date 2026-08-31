# 18 — Perchance AppImage Integration

## Objective
Make the unused Perchance image/text/gallery tabs functional through an optional Linux AppImage/service integration.

## Implementation
1. Identify the exact Perchance executable/API expected by the tabs and verify its license/redistribution terms.
2. Prefer a user-approved download or user-provided binary if direct bundling is not permitted.
3. Add AppImage discovery, executable validation, version probing, and optional install/update flow.
4. Launch it as a managed child process with a fixed/negotiated local port, health check, log capture, and parent-death cleanup.
5. Build a provider adapter so UI tabs call Easy Diffusion's backend API instead of directly depending on process details.
6. Map image/text/gallery operations explicitly and provide unavailable/offline UI state.
7. Sandbox filesystem/network access as much as practical and do not expose arbitrary command-line execution through the UI.

## Dependencies
Linux process supervisor pattern; Windows counterpart is a separate objective.

## Validation
Missing AppImage, first install, successful start, health failure, restart, tab operations, gallery persistence, shutdown, and upgrade.

## Complete when
The Perchance tabs are functional when the optional backend is present and degrade cleanly when it is absent.
