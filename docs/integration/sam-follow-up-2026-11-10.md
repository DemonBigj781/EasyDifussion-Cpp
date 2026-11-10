# SAM follow-up: confirmed adapter bug and remaining gaps

User reports SAM does not work. Prior passing contracts were not evidence of
working model inference.

## Confirmed bug, fixed locally

The bundled helper's sam_write_masks() treats --out as a prefix: single-mask
output is prefix + ".png"; multimask output is prefix + index + ".png".
The Python adapter instead created a PNG placeholder, passed its complete name,
then read that empty placeholder. The real output was a different file and was
not cleaned up. Its mock test repeated the incorrect filename assumption.

The local correction uses a private temporary directory, reads actual single
or numbered outputs, validates PNG dimensions, returns the first available
mask plus all masks for multimask requests, and cleans up all outputs. It
reports a clear error if the helper's quality filters produce no files.

## Build and format safeguards, also fixed locally

- source/Build.sh defaults to building the sdkit target. The sam helper is a
  separate CMake target, with no dependency ensuring the ordinary sdkit build
  creates it. The local fix adds add_dependencies(sdkit sam); Ninja confirms
  bin/sam is now a dependency of bin/sdkit.
- The model catalog exposes .bin, .gguf and .pth, but this bundled helper checks
  legacy GGML magic 0x67676d6c. The local adapter now checks only the first four
  bytes and reports a clear conversion error for incompatible formats. It does
  not hide catalog entries or claim a signature check validates all weights.

## Remaining runtime and UX limitations

- Both UI scripts display a generated mask preview. These tests do not establish
  a complete interactive selection-to-inpainting workflow.
- The /sam/segment handler inspected is the Python server handler, not proof of
  a completed native Cosmopolitan SAM route.

## Checks

Eight standalone SAM contracts pass with realistic output-prefix behavior,
multi-mask filtering/cleanup and missing-output cases. Twenty-one combined native
image-tool/color-inpaint/SAM unittest cases pass. The separate SAM catalog
script also passes. These use synthetic masks and a mocked helper, not actual
segmentation. The real SAM helper also compiled successfully in an explicitly
CPU-only CMake build, and its --help command ran successfully. No full sdkit build,
model inference or GPU execution was performed; actual model inference
and both live UI flows remain required.
