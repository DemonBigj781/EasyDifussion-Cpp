# Superseded staged CUDA prototype

The `cuda-*.cu` stage kernels in this directory predate the registered Common
CUDA translation. They materialize a score buffer, use the retired `easyapi`
feature namespace, and do not implement the complete normalized tensor, mask,
bias, or head-broadcasting contract.

They are retained only as reference code. No production CMake or test target may
compile them. The authoritative Common CUDA route is
`../cuda/translation/gpu/forward.cu`, which implements the same logical QKT,
mask, softmax, and AV stages in one fused operation.
