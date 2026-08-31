# 28 — Tangent101 Easy-Diffusion-Plugins-1 `patch-14` Integration

## Objective
Integrate useful plugins from Tangent101/Easy-Diffusion-Plugins-1 patch-14 without blindly copying incompatible or duplicate legacy code.

## Implementation
1. Pin the exact upstream patch-14 commit/ref and record license/provenance for each imported plugin.
2. Inventory plugins and classify: already implemented, compatible as-is, needs adapter, conflicts with bundled plugin, obsolete, or unsafe/unmaintained.
3. Import one plugin at a time through the central plugin registry with stable IDs and version metadata.
4. Port old API/DOM assumptions through compatibility adapters rather than modifying the modern UI globally.
5. Detect duplicates against existing bundled/modern plugins and select one authoritative implementation.
6. Preserve attribution and third-party notices.
7. Add a smoke test per accepted plugin and explicitly list intentionally-not-integrated plugins with reason.

## Dependencies
Legacy-plugin registry/deduplication and per-plugin controls.

## Validation
Plugin discovery, duplicate cases, each accepted plugin's core action, failure isolation, disable/re-enable, clean uninstall/removal, and UI restart.

## Complete when
Every patch-14 plugin has a documented disposition and all accepted plugins run through the unified plugin system without duplicate registration.
