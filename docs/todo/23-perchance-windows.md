# 23 — Perchance Windows Build

## Objective
Provide the Perchance integration on native Windows without AppImage, WSL, or Wine assumptions.

## Implementation
1. Complete the base Windows support first: process management, networking, paths, packaging, and shutdown.
2. Determine the Windows build/package format for the exact Perchance component used by the Linux integration.
3. Verify redistribution/license requirements independently of the Linux AppImage.
4. Add executable discovery/install/update and version probing.
5. Reuse the same provider adapter/API used by the Linux integration so the UI is platform-neutral.
6. Implement Windows child-process lifecycle, health checks, log capture, port selection, and cleanup.
7. Handle antivirus/file-lock and user-directory path issues without requiring administrator privileges where possible.

## Dependencies
`14-windows-support.md` and `18-perchance-appimage.md` provider abstraction.

## Validation
Fresh Windows install, missing backend, install/update, service start/restart, all Perchance tabs, path spaces, shutdown, and uninstall/cleanup.

## Complete when
The same Perchance-backed features work natively on Windows through the shared Easy Diffusion adapter.
