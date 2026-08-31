# 17 — Pipeline UI Arrangement Menu

## Objective
Let users rearrange, show, and hide pipeline UI sections without changing the actual execution graph accidentally.

## Implementation
1. Give every movable panel a stable ID and metadata describing title, group, default position, and plugin ownership.
2. Store visual ordering in user settings, not only DOM order.
3. Add drag/drop plus keyboard-accessible move up/down controls.
4. Add hide/show toggles and reset-to-default.
5. Keep execution dependencies separate from visual layout: moving a ControlNet panel must not reorder backend execution.
6. Provide plugin registration hooks so plugin-created panels participate without editing core layout code.
7. Version the saved layout schema so renamed/removed panels do not corrupt settings.
8. On startup, merge saved order with newly-added panels deterministically.

## Dependencies
Per-plugin controls and legacy plugin modernization may share the same stable plugin/panel IDs.

## Validation
Reordering, persistence across restart, hidden panels, reset, keyboard use, newly-installed plugin panels, removed plugins, mobile/narrow layout, and old settings migration.

## Complete when
Users can customize and persist the UI arrangement safely, and visual customization never changes generation semantics.
