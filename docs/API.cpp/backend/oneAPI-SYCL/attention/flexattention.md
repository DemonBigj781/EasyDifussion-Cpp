# FlexAttention — oneAPI / SYCL

## Status

**Backend-neutral host implementation.**

This repository's FlexAttention feature is the native C++ high-resolution visual-token-selection policy. It is not PyTorch's generic FlexAttention API. The current implementation operates on host `float` attention data and produces a host selection mask.

It can therefore be used alongside a SYCL execution path when the required attention data is made available to the host, but there is no SYCL-native selection kernel in the current implementation.

## Work required for device-native execution

A SYCL implementation would need device-side reduction, normalization, thresholding, adaptive max pooling, mask expansion, and a defined transfer/synchronization policy.