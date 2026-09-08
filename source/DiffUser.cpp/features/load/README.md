# Load feature layout

Load routes are organized by backend first and tensor/resource type second:

```text
features/load/
├── common/devices/
│   ├── load.hpp
│   └── load.cpp
└── [backend]/
    ├── definition/[device]/[tensor_type].cpp
    └── translation/[device]/[tensor_type].cpp
```

The definition file owns native preparation. Its matching translation adapts
and registers it with the final `common/devices` contract. Common presents the
same API regardless of originating backend or device. Tensor types include `model.cpp`, `vae.cpp`,
`unet.cpp`, `clip.cpp`, `clip-vision.cpp`, `condition.cpp`, `image.cpp`,
`latent.cpp`, and `mask.cpp` as they become implemented.

Currently only the CPU/CPU and CUDA/GPU `model.cpp` pairs are implemented. The
remaining tensor/resource types stay absent until their Common contracts,
backend behavior, and validation exist.
