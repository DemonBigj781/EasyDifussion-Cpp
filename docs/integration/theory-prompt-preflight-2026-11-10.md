# Theory conditioning rule in the active application

## Production change

Theory's conditioning parser (source/INFERENCE.cpp/features/conditioning/common/
prompt_attention.cpp at commit 4755b7abe48b49207e33bbff786dc3e2a7032491)
rejects non-finite attention weights. This increment
applies that safety rule to the active image/video API before task creation or
model loading. Both prompt and negative_prompt are checked. Invalid numeric
weights, out-of-range conversions and overflowing products return HTTP 400 with
a field-specific message. Missing/empty prompts and finite zero/negative weights
remain valid; accepted text is not rewritten.

The existing application's parser remains authoritative. Theory's parser is not
a drop-in replacement: its shorthand, bare-colon, fusion-group and AND semantics
differ from current rendering. Those changes are deliberately not activated.
This is a behavior-derived integration, not a wholesale Theory source import.

An application-owned runtime_adapters translation unit is added to the existing
stable-diffusion library from top-level CMake. It exports a narrow SD_API C ABI
using pointer/length input and status codes. The internal parser and exceptions
stay inside the runtime. JSON interpretation and error messages stay in the
application. This avoids exporting internal C++ containers on Windows, modifying
vendor files, duplicating a parser or allocating another model runtime.
Unexpected adapter failures remain server errors rather than HTTP 400.

## Verification and limits

The CPU application executable and seven optional native contracts are tested.
The prompt contract covers ordinary weights, malformed numbers, explicit and
nested overflow, both fields, escapes, BREAK, Unicode, literal shorthand and
request types. The optional scripts/test_prompt_preflight.py launches a CPU-only
server in an empty temporary working directory and sends 12 invalid txt2img and
img2img requests. Each must return 400 and leave its task absent (progress 404).

Run the HTTP check with SDKIT_TEST_BINARY pointing at a CPU-only sdkit executable.
The test does not submit executable generation requests, enumerate devices or
load models. Shared video handlers receive the same preflight; video generation
and video LoRAs are not tested.

Theory's tokenizer, text-encoder execution, fusion policies and broader
model-resource migration remain unintegrated. No pretrained inference or GPU
parity is claimed. Windows export uses the existing SD_API mechanism, but a
Windows build is not available in this environment.

Validation on 2026-11-10: the final CPU Release executable built successfully;
all seven native contracts and the live HTTP regression passed. Another 35
existing prompt/UI/image/SAM/inpainting unit contracts passed. The prompt test
is registered in ordinary SDKIT_BUILD_TESTS builds as well as the optional
application-contract configuration. Null input and embedded NUL handling are
covered at the adapter boundary. The ELF shared-library export was verified.
The final runtime adapter and all seven contracts also passed ASan/UBSan.
An initial sanitizer rebuild's compiler was killed while another runtime build
was active; a single-job retry after that build completed succeeded. LeakSanitizer
remains disabled because the executor's ptrace environment prevents it.
