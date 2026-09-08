# TeaCache — oneAPI / SYCL

## Status

**Native policy; host-resident cache state.**

TeaCache is a backend-neutral condition-boundary implementation. It stores previous diffusion input and full-model residual data in host memory and applies the model-specific polynomial rescaling policy there. There is no SYCL-specific TeaCache kernel or device-resident residual cache in the current implementation.

A future SYCL optimization may reduce transfers or keep residual state on device, but generic TeaCache policy remains outside the backend layer.