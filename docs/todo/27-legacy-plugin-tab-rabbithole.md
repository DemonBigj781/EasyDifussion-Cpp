# 27 — Repair Legacy Plugin Tab and Preserve RabbitHole in the 3.5 UI

## Objective
Restore the legacy plugin tab enough for old 3.5-era plugins to remain usable, specifically ensuring RabbitHole continues to work while the modern UI evolves.

## Implementation
1. Reproduce the current failure with browser console/network logs and identify whether it is tab registration, asset loading, legacy globals, routing, or backend API mismatch.
2. Define a legacy compatibility shell that supplies only the APIs old plugins actually need.
3. Keep the legacy tab isolated from modern routing/state so old scripts cannot overwrite current UI initialization.
4. Add RabbitHole as a regression fixture: document its required globals, events, endpoints, DOM hooks, and storage behavior.
5. Route old backend calls through compatibility adapters when endpoint names or payloads changed.
6. Prevent duplicate plugin loading through the central plugin registry.
7. Keep clear deprecation boundaries: compatibility layer fixes should not force new plugins to use old APIs.

## Dependencies
Legacy plugin registry/deduplication should be developed alongside this work.

## Validation
Open/close legacy tab, reload page, load RabbitHole, execute its core actions, coexist with modern plugins, handle one failing legacy plugin, and preserve browser history/routing.

## Complete when
The legacy tab loads reliably and RabbitHole's documented core workflow passes a repeatable regression test in the old 3.5-compatible surface.
