# SageAttention

The current CUDA slice normalizes capability selection without overstating the
forward route:

```text
stable-diffusion.cpp GGML selector
  -> Sage Common support
  -> CUDA support translation
  -> CUDA-native support definition
  -> direct GGML-native SM80 forward compatibility kernel
```

`common/sage_attention.*` owns the backend-neutral tensor, device, capability,
validation, registry, and support-dispatch contract. The CUDA definition
records the live kernel boundary: SM8x, F32 query/output, F16 key/value, head
dimensions 64 or 128, compatible grouped-query shapes, GGML head-major output
strides, and no mask, attention sink, bias, or logit soft-cap support. The CUDA
translation registers that definition with Common.

`source/API.test/Feature/Attention/Sage/Cuda/Support.cpp` is a build-only
contract test. It does not execute CUDA or establish numerical/runtime proof by
itself.

Manual runtime validation on an RTX 3060 (`sm_86`, driver 580.94.18) completed
a profiled one-step 256x256 SDXL Turbo generation and wrote a valid RGB PNG.
The Nsight Systems trace recorded 280 launches each of the Sage key-mean, F16 K
quantization, F32 Q quantization, INT8-QK/FP16-PV attention, and output
conversion kernels. This is runtime proof of the Common support selector and
retained native forward path, not numerical parity proof or a normalized Common
`forward` route.

The next migration step is a normalized Common `forward` request and CUDA
translation. Once that exists, the GGML-to-Common request adapter belongs in
`source/API.bridge/sdkit3-ggml`, and the direct native launch can be retired
after application and fallback coverage are proven.
