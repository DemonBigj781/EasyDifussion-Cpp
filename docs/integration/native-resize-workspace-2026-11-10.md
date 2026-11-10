# Checked native resize workspace

The previous raw image-buffer check did not bound stb's internal scratch
allocations. For example, a 2x2 RGB source resized to 100,000,000x1 passes the
raw destination-byte bound but overflows the signed ring-buffer size. This
increment rejects such requests before allocating the destination or reading
source pixels.

The application wrapper now mirrors the bundled default-filter workspace
layout using checked arithmetic. It retains stb's float scale/support/ceil
operations, validates filter widths before integer conversion, checks each
coefficient/contributor/decode/horizontal/ring/encode buffer and checks their
signed-int sum. Even fields that stb computes before zeroing are checked.
No vendor file, image filter or normal resize rounding is changed.

This is deliberately coupled to the included stb implementation. Review the
preflight alongside any future stb update rather than assuming the scratch
layout is a stable public interface. The source's unsigned dimensions have
already passed the raw-buffer checks before entering this calculation.

Regression cases exercise ring-buffer overflow, individually representable
buffers whose sum overflows, and extreme downsampling that would overflow the
filter-width cast. They use tiny allocations or sentinel descriptors and must
fail before large allocation/pixel access, preserving the original descriptor.
Ordinary up/down/mixed-scale resizes and existing PNG round trips exercise all
four channel formats with unchanged solid pixels.

This fixes resize workspace representability, not all bundled image-library
issues. PNG filter-score/compression arithmetic at extreme sizes and allocator
failure behavior inside vendor code remain separate work. It does not establish
arbitrary-size image support or model/GPU inference readiness.

Validation on 2026-11-10: the full CPU application built; nine native contracts
passed in Release and ASan/UBSan. The live API regression and 35 existing
prompt/image/SAM/inpainting unit tests passed. Source review confirmed the
workspace layout and checked intermediate arithmetic. LeakSanitizer remains
disabled under executor ptrace. No model/GPU/video-LoRA testing or Media-drive
inspection occurred; Windows and a complete Cosmocc artifact remain untested.
