# Exception-safe ownership in native image generation

## Concrete application corrections

The existing image path held decoded buffers in raw descriptors and freed them
only after generate_image returned normally. A malformed mask could throw before
its already-decoded init image was registered for cleanup. Later ControlNet,
IP-Adapter or reference-image errors could leak all earlier inputs. Reference
vector insertion and result encoding also had allocation-exception leak paths.
A throwing native generation/callback could leave active_generation_ctx_ set.

This increment applies the ownership guarantees established during Theory
review to the real application resources, without replacing its model manager:

- Scoped input ownership is installed immediately after generation parameters
  are initialized. Each decoded input is published before the next risky step.
- Reference construction guards both accumulated images and the current image
  until ownership transfers successfully to the vector.
- The result-array guard exists before the native call can publish output and
  frees partial output on native-call or encoding exceptions.
- Empty image-encoding results become an explicit request error instead of a
  successful response containing an empty image string.
- Request-state cleanup clears the active context and this generator's callback
  pointers while its request mutex remains held, including on exceptions.
- Inputs, outputs and state are explicitly reset before recursive CPU fallback
  unlocks the request mutex. Old scopes cannot retain buffers or clear the
  retry's newly published callback state.

These are guards around existing image descriptors, not a second tensor/model
runtime. Sampling, successful image output, inpainting semantics, CPU fallback
policy and both UIs are unchanged. Video ownership paths are not modified.

## Validation scope

The ownership contract uses counted allocations and the actual guard types.
It tests failures after each input/reference allocation, reference-list transfer,
partial native output, output-encoding failure, idempotent cleanup, and state
cleanup before a simulated retry. A separate real PNG decode test supplies an
invalid mask after decoding the init image and verifies that scoped cleanup
clears both descriptors. Existing resize/PNG round trips continue to run.

These fixtures do not run a pretrained diffusion model or force failures inside
actual hardware generation. The generation function compiles in the full CPU
application; resource failure injection targets its shared cleanup mechanisms.
Global callback ownership across multiple simultaneous generator instances and
the separate video path remain broader work. No GPU/video-LoRA tests are run.

Validation on 2026-11-10: full native CPU application built; nine native contracts
passed in Release and ASan/UBSan, including counted cleanup and real PNG/mask
failure fixtures. The live API prompt regression, 120 foundation tests and 35
existing prompt/image/SAM/inpainting unit tests passed. LeakSanitizer remains
disabled under executor ptrace; explicit allocation accounting tests cleanup.
No pretrained model, Media drive or prohibited hardware was used. Windows and
complete Cosmocc artifact validation remain pending.
