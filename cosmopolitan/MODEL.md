# Embedded inference fixture

The portable application embeds `stories260K.gguf` to verify real model loading,
tokenization, prompt evaluation and decoding without another runtime or model
download. This tiny model is a development fixture, not a replacement for an
image-generation checkpoint or a useful general-purpose language assistant.

- GGUF publisher: [ggml-org/tiny-llamas](https://huggingface.co/ggml-org/tiny-llamas).
- Exact file: [revision def3e2dd70df35ecbf6403ea347de4c5977220c1](https://huggingface.co/ggml-org/tiny-llamas/blob/def3e2dd70df35ecbf6403ea347de4c5977220c1/stories260K.gguf).
- SHA-256: `047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b`.
- Original model family: [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas), Llama 2 architecture trained on TinyStories.
- Original model license: MIT, as recorded in the model card. The accompanying
  [llama2.c license](https://github.com/karpathy/llama2.c/blob/master/LICENSE)
  is retained in `licenses/tinyllamas.LICENSE` and inside the executable.

The build fetches the exact revision and verifies its digest before embedding
it. Model weights are not committed to this repository. User-selected models
remain data files; the one-executable requirement concerns the application and
its executable code.
