# Batch Background Remover integration

Easy Diffusion's U²-Net ONNX background-removal workflow is based on the
batch-processing approach demonstrated by
[`gohard-lab/batch_bg_remover`](https://github.com/gohard-lab/batch_bg_remover).
The runtime implementation in this repository is native Easy Diffusion code:
it calls ONNX Runtime directly and does not copy the upstream Python sources.

The U²-Net model is stored at
`source/models/remove_background_onnx/u2net.onnx`. U²-Net is distributed under
the Apache License 2.0; see the
[`xuebinqin/U-2-Net`](https://github.com/xuebinqin/U-2-Net) project for the
original model and license.
