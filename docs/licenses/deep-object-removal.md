# Deep Object Removal model attribution

The optional `deep-object-removal-400.onnx` runtime model is converted from
VPanjeta/Deep-Object-Removal's `model/pretrained_model` checkpoint and preserves
the original 400×400 reconstruction network. The upstream source and model are
licensed under Apache License 2.0:

- https://github.com/VPanjeta/Deep-Object-Removal
- Copyright the upstream Deep-Object-Removal contributors

Easy Diffusion does not require TensorFlow at runtime. The conversion script is
`scripts/convert_deep_object_removal.py`; inference uses the existing ONNX
Runtime dependency.
