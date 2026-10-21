# SAM Segmentation UI Design

## Goal

Expose the repository's existing Segment Anything executable through matching
segmentation panels in the C++ and legacy web UIs. The feature accepts an
image plus one point or box prompt and returns a PNG mask for preview or
subsequent editor use.

## Architecture

Both UIs call one existing-host API route, `POST /sam/segment`. The route
follows the repository's native image-tool pattern: validate a bounded base64
image request, resolve an installed SAM model and helper executable, write
temporary input/output files, invoke the helper without a shell, and return a
PNG data URL with the source dimensions.

The current SAM implementation is a monolithic standalone executable. This
design deliberately reuses it rather than copying its model implementation
into the main C++ server. That keeps the request contract shared and limits
the change to the existing helper-process integration boundary.

## Request contract

```json
{
  "image": "data:image/png;base64,...",
  "model": "/absolute/or/approved/model/path",
  "prompt_type": "point",
  "point": {"x": 120, "y": 80},
  "box": {"x1": 20, "y1": 20, "x2": 400, "y2": 300},
  "multimask": false
}
```


Exactly one prompt type is required. Coordinates are validated as finite,
non-negative values and are checked against the decoded image dimensions.
Model resolution is restricted to known SAM model directories or the explicit
helper configuration used by the existing native-tool conventions.

## UI behavior

Each surface provides:

- source image file input and preview;
- SAM model selector;
- point or box prompt mode;
- coordinate fields appropriate to the selected mode;
- a Generate mask action;
- busy/error status and returned mask preview.

The C++ UI renders the panel on the main page and loads a dedicated script.
The legacy UI loads an equivalent optional plugin. Both scripts serialize the
same request shape and render the same response shape. AnimateDiff controls
remain unchanged because they are already implemented and verified.

## Failure handling

Invalid input returns HTTP 400 with a user-readable detail. Missing models or
the helper return a controlled installation error. Helper timeouts return HTTP
504. Other helper failures return HTTP 500 without exposing unbounded process
output. Requests are serialized to avoid overlapping large SAM model loads.

## Verification

- Python contract tests cover request validation, prompt serialization, helper
  argument construction, and response encoding.
- C++ UI tests cover panel rendering and successful request/response wiring.
- Legacy UI tests cover panel rendering, mode switching, request payloads,
  response display, and responsive layout.
- The SAM target is built with one compiler worker and the relevant test suite
  is run afterward.
