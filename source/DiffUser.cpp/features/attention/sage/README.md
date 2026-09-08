# SageAttention

The current source normalizes capability selection without overstating a
forward route. DiffUser does not depend on stable-diffusion.cpp, GGML, SDKIT3,
or llama.cpp.

`common/sage_attention.*` owns the backend-neutral tensor, device, capability,
validation, registry, and support-dispatch contract. The CUDA definition
records the live kernel boundary: SM8x, F32 query/output, F16 key/value, head
dimensions 64 or 128, compatible grouped-query shapes, GGML head-major output
strides, and no mask, attention sink, bias, or logit soft-cap support. The CUDA
translation registers that definition with Common.

`source/API.test/Feature/Attention/Sage/Cuda/Support.cpp` is a build-only
contract test. It does not execute CUDA or establish numerical/runtime proof by
itself.

There is no Sage CPU definition, CPU translation, CPU test, or CPU forward
implementation. The CPU family is therefore not accommodated. CUDA forward
work also remains separate from its existing support contract.
