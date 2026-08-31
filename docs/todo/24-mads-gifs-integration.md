# 24 — mads-gifs / gif.js / gif.worker.js Integration

## Objective
Integrate GIF creation into the current UI using the bundled mads-gifs plugin behavior plus `gif.js`/`gif.worker.js`, without requiring the old plugin UI.

## Implementation
1. Inventory the plugin's useful behavior and separate it from legacy DOM assumptions.
2. Vendor or reference `gif.js` and `gif.worker.js` according to their license and existing repository policy.
3. Create a modern GIF panel/action with frame source selection, ordering, delay/FPS, loop count, quality, dimensions, and output name.
4. Run encoding in a worker so the UI remains responsive.
5. Add progress, cancellation, memory/error handling, and maximum-frame/size safeguards.
6. Support frames from generated-image history/gallery and uploaded files.
7. Save output through the existing artifact/gallery path and attach generation/source metadata where useful.
8. Avoid duplicate registration if the legacy mads-gifs plugin is also discovered.

## Dependencies
Legacy-plugin deduplication and stable gallery APIs are helpful but not mandatory for a first native integration.

## Validation
Different frame sizes, transparency, loop settings, long jobs, cancellation, worker load failure, mobile browser behavior, and gallery save/reload.

## Complete when
Users can create and save GIFs from the current UI without opening the legacy plugin interface.
