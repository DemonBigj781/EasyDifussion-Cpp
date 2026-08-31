# xFormers — oneAPI / SYCL

## Status

**Not implemented.**

The common xFormers ABI currently reports unsupported for every backend. There is no SYCL-specific xFormers kernel wired into API.cpp.

## Work required

Implement a backend-appropriate native kernel/algorithm, expose capability probing through the common attention API, and validate correctness/performance on supported SYCL devices.