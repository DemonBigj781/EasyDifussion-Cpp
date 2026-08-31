# EasyDifussion-Cpp TODO

Every objective has a standalone implementation document under `docs/todo/`. The checkbox here is the project-level status; the linked document defines the implementation approach, dependencies, validation, fallback behavior, and completion criteria.

## Runtime / architecture
- [ ] [01 — Migrate Python to 3.13](docs/todo/01-python-3.13-migration.md)
- [ ] [02 — Additional video model compatibility](docs/todo/02-video-model-compatibility.md)
- [ ] [03 — YOLO integration](docs/todo/03-yolo-integration.md)
- [ ] [04 — Pascal/Volta/Turing compatible Flash Attention](docs/todo/04-legacy-nvidia-flash-attention.md)
- [ ] [05 — aarch64 support](docs/todo/05-aarch64-support.md)
- [ ] [06 — ROCm support](docs/todo/06-rocm-support.md)
- [ ] [07 — IPEX support](docs/todo/07-ipex-support.md)
- [ ] [08 — oneAPI/SYCL support](docs/todo/08-oneapi-sycl-support.md)
- [ ] [09 — OpenVINO support](docs/todo/09-openvino-support.md)
- [ ] [10 — OpenCL support](docs/todo/10-opencl-support.md)
- [ ] [11 — OpenGL compute support](docs/todo/11-opengl-compute-support.md)
- [ ] [12 — JetPack v5 support](docs/todo/12-jetpack-v5-support.md)
- [ ] [13 — JetPack v6 support](docs/todo/13-jetpack-v6-support.md)
- [ ] [14 — Windows support](docs/todo/14-windows-support.md)
- [ ] [15 — Wine/Proton exclusive RAM offloading](docs/todo/15-wine-proton-offloading.md)
- [ ] [16 — Multi-GPU K-sampling](docs/todo/16-multi-gpu-k-sampling.md)

## UI / services / ecosystem
- [ ] [17 — Pipeline UI arrangement menu](docs/todo/17-pipeline-ui-arrangement.md)
- [ ] [18 — Perchance AppImage integration](docs/todo/18-perchance-appimage.md)
- [ ] [19 — Stable Horde client API/UI](docs/todo/19-stable-horde-client.md)
- [ ] [20 — Complete Civitai/Hugging Face API/UI](docs/todo/20-civitai-huggingface-ui.md)
- [ ] [21 — OpenRouter client UI](docs/todo/21-openrouter-client-ui.md)
- [ ] [22 — MCP API](docs/todo/22-mcp-api.md)
- [ ] [23 — Perchance Windows build](docs/todo/23-perchance-windows.md)
- [ ] [24 — mads-gifs / gif.js integration](docs/todo/24-mads-gifs-integration.md)
- [ ] [25 — Legacy plugin integration and deduplication](docs/todo/25-legacy-plugin-integration.md)
- [ ] [26 — Spellcheck/tokenizer CSV integration](docs/todo/26-spellcheck-tokenizer-csv.md)
- [ ] [27 — Legacy plugin tab + RabbitHole repair](docs/todo/27-legacy-plugin-tab-rabbithole.md)
- [ ] [28 — Tangent101 patch-14 plugin integration](docs/todo/28-tangent101-plugin-integration.md)
- [ ] [29 — Per-plugin enable/disable controls](docs/todo/29-plugin-controls.md)
- [ ] [30 — Live runtime argument injection/reload](docs/todo/30-live-runtime-arguments.md)

## Hardware validation / deep integration
- [ ] [31 — Validate oneAPI/SYCL on H3C XG310](docs/todo/31-xg310-oneapi-validation.md)
- [ ] [32 — Merge llama.cpp into stable-diffusion.cpp safely](docs/todo/32-merge-llama-stable-diffusion.md)

## Related architecture notes
- [Native Hugging Face + LoRA to GGUF plan](docs/native-hf-lora-gguf-plan.md)
- [Easy Diffusion oneAPI/SYCL expansion plan](docs/easy-diffusion-oneapi-plan.md)

A checklist item should only be marked complete after the acceptance criteria in its own document are satisfied.
