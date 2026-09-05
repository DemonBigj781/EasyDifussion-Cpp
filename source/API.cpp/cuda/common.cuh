#pragma once

// Shared ggml CUDA primitives are not attention-owned. Centralize the bridge
// here so API.cpp implementations do not reach into the legacy tree directly.
#include "../../sdkit3-port-source/stable-diffusion.cpp/ggml/src/ggml-cuda/common.cuh"
