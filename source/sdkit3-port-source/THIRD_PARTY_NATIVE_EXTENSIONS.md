# Native extension provenance

This port contains clean C++ implementations informed by the following
research/reference projects. No Python runtime is embedded by these features.

| Feature | Reference | Pinned revision | License | Implementation note |
| --- | --- | --- | --- | --- |
| IP-Adapter | https://github.com/tencent-ailab/IP-Adapter | `62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2` | Apache-2.0 | Native SD1.x/SDXL base projection and cross-attention K/V injection. |
| FlexAttention token selection | https://github.com/UMass-Embodied-AGI/FlexAttention | `f814be5187e1ae714c8eb8161fcc599c983c3be5` | Apache-2.0 | Native high-resolution visual-token selection policy; it is not presented as PyTorch's unrelated generic attention API. |
| TeaCache | https://github.com/ali-vilab/TeaCache | `7c10efc4702c6b619f47805f7abe4a7a08085aa0` | Apache-2.0 | Native residual reuse policy and published model-specific polynomial rescaling. The streamed-weight backend measures the diffusion input at the condition boundary instead of constructing a second graph solely to observe the first modulated block. |
| EasyCache | https://github.com/H-EmbodVis/EasyCache | `bfc63232cfae8c77e2dccf9d72b8f2f56bd995b8` | Apache-2.0 | Native reuse dispatcher integrated with the sampler cache lifecycle. |
| Ultralytics-compatible YOLO inference | https://github.com/ultralytics/ultralytics | `94aec674e24efcd953c441c976a3b76fb4373865` | AGPL-3.0 | The runtime sidecar is a clean C++ TorchScript loader, preprocessor, decoder, and NMS implementation. Ultralytics is used only for one-time model export; it is not imported at runtime. Exported model artifacts retain their applicable license and metadata. |
| Text-removal workflow | https://github.com/thatguyshzr/remove_text_from_image | `ae8b109ba449098fffc1e743a37c6378b823ca46` | No license declared | No source was copied. The native helper is a clean-room high-contrast component/line detector that produces an editable diffusion inpaint mask instead of embedding the reference project's Keras OCR/OpenCV code. |

Model weights retain their own licenses and are not redistributed by these
source changes.
