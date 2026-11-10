# Image-buffer arithmetic in the active application

## Reconciliation with Theory

Theory's request plan at 4755b7abe48b49207e33bbff786dc3e2a7032491 validates
non-zero dimensions, finite guidance and strength. Those draft policies are not
a drop-in application contract: main supports finite negative CFG, model-specific
size alignment, ControlNet nearest-64 rounding and strength above one. This
increment leaves those behaviors unchanged and fixes concrete image allocation
arithmetic at the layer that knows the final sizes and channel count.

This is a production correction informed by the ownership/validation work, not
an import of the unfinished Theory inference executor.

## Corrected paths

- Resize validates source descriptors and final target byte counts before
  signed-int multiplication or narrowing into the bundled stb API. Allocation
  uses the checked size_t count. Failed resize leaves source ownership intact.
- PNG encoding validates dimensions, channels, stride and the extra per-row
  PNG filter byte against the downstream signed-int range before reading pixels.
- Decoding checks channel policy and encoded input length before int narrowing,
  and verifies the returned image descriptor before ownership is published.
- Default img2img masks use a checked solid-image allocator instead of raw
  signed width*height multiplication in malloc and memset. Successful masks
  remain all white with one channel, exactly as before.

The bounds follow the existing stb signed-int interfaces; no new arbitrary
resolution cap is introduced. Channels 1–4 and the existing clamp-to-eight
resize behavior are preserved. No vendor or third-party files are changed.

## Test scope

The new image-buffer-bounds test uses actual image_utils implementation and
real stb resize/PNG encode/decode. Grayscale, grayscale-alpha, RGB and RGBA solid fixtures resize
and round-trip losslessly. Tests also cover no-op ownership, failed-resize
ownership, default white masks, invalid channel policies, unsigned dimensions
outside int range, overflowing pixel products and PNG row-filter overflow.
Invalid large descriptors contain only a single sentinel pixel: they must be
rejected before pixel access or any large allocation.

This is not end-to-end model inference or a complete audit of stb's scratch
allocation internals. General API numeric normalization and model-specific
allocation arithmetic remain separate work. Theory tokenizer/conditioning
execution and GPU resource migration remain incomplete. Windows is untested.

Known residual limits: the bundled stb resize implementation also sizes filter,
coefficient and ring-buffer scratch allocations with signed ints; raw pixel
bounds do not prove those calculations safe for extreme aspect/scale ratios.
PNG filter scoring and compression output arithmetic have their own limits.
These pre-existing vendor-internal issues are not fixed by this raw-buffer
increment. Do not treat INT_MAX raw bytes as a guaranteed supported image size.

Validation on 2026-11-10: full CPU application build succeeded; eight native
contracts passed in Release and ASan/UBSan, including the real image operations.
All four channel formats and automatic decoder channels passed. The live HTTP
prompt rejection test and 35 existing prompt/image/SAM/inpainting regressions
passed. LeakSanitizer remains disabled under the executor's ptrace environment.
No models, GPUs, Media drives or video LoRAs were used.
