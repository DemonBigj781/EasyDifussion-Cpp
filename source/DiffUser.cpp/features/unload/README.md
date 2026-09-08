# Unload feature layout

Unload routes are organized by backend, device class, and tensor/resource type:

```text
features/unload/
├── common/devices/
│   ├── unload.hpp
│   └── unload.cpp
└── [backend]/
    └── [device]/
        ├── definition/[tensor_type].cpp
        └── translation/[tensor_type].cpp
```

The definition releases the native resource owned by that exact backend and
device class. Its matching translation registers through final
`common/devices`, which exposes one identical unload API for every origin. Missing files
mean unsupported.

Currently only the CPU/CPU and CUDA/GPU `model.cpp` pairs are implemented.
