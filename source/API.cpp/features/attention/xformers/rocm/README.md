# ROCm xFormers methods

Structural placeholder for feature-first xFormers ROCm/HIP method implementations.

Canonical method path: `API.cpp/features/attention/xformers/rocm/definition/[device]/[method].cpp`.
ROCm-to-common adapters live under `rocm/translation/[device]/`.

ROCm method files should mirror the common xFormers method contract without changing CUDA or existing API implementations. Runtime support remains unvalidated until tested on suitable AMD hardware.
