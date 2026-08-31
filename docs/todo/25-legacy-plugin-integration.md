# 25 — Legacy Plugin Integration and Duplicate Resolution

## Objective
Integrate bundled legacy UI plugins into the current UI while preventing duplicate plugin loading and conflicting registrations.

## Implementation
1. Create a plugin manifest model with stable plugin ID, source, version, entry point, UI surfaces, dependencies, and compatibility range.
2. Scan bundled/current/legacy plugin locations into one registry before executing plugin code.
3. Deduplicate primarily by explicit plugin ID; use normalized source/name hashes only for legacy plugins without IDs.
4. Define precedence: modern built-in implementation > explicitly-selected installed plugin > bundled legacy copy, unless user override is supported.
5. Prevent duplicate script injection, routes, tabs, event handlers, and background workers.
6. Add compatibility shims for old Easy Diffusion plugin APIs rather than copying old globals into the entire new UI.
7. Record disabled/conflicting plugins and reason in diagnostics.
8. Provide per-plugin controls through the dedicated plugin-control objective.

## Validation
Same plugin in two locations, two versions, alias names, incompatible legacy plugin, plugin throwing during load, duplicate tab/route prevention, and restart persistence.

## Complete when
Bundled legacy plugins can be discovered and loaded through one registry with deterministic conflict resolution and no duplicate UI/runtime behavior.
