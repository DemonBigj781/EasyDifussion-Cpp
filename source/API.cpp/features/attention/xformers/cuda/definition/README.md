# CUDA GPU definition

`gpu/xformers.hpp` declares the CUDA-native metadata contract used below Common. The five method files validate their corresponding semantic portions of a fused request:

- `qkt.cpp`: Q/K pointers, dtypes, byte strides, extents, batch broadcasting, dimensions, and divisible GQA/MQA heads;
- `mask.cpp`: scale, soft-cap, supported mask dtypes/broadcasting, explicit or maximum-bias ALiBi, and attention sinks;
- `softmax.cpp`: fused row-count launch limits;
- `av.cpp`: V/output pointers, extents, head mapping, output shape, and the 512-value-dimension limit;
- `forward.cpp`: composed validation and advertised CUDA capabilities.

Definitions contain no Common registration and perform no launch. `translation/gpu/forward.cu` is the only Common adapter and fused-launch owner.
