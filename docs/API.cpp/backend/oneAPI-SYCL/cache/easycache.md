# EasyCache — oneAPI / SYCL

## Status

**Native policy; host-resident cache state.**

EasyCache is implemented in backend-neutral C++ and stores condition diffs, previous input/output values, and transformation-rate state in host `std::vector<float>` containers. There is no SYCL-specific EasyCache kernel or device-resident cache implementation.

When SYCL executes the model, EasyCache remains a sampler/runtime policy. API.cpp backend work should cover only SYCL-specific transfers, synchronization, or future device-resident acceleration—not move the generic EasyCache algorithm into the backend.