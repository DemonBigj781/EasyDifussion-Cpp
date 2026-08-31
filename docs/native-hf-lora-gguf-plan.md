# Native Hugging Face + LoRA to GGUF Conversion Plan

## Status

Design notes / implementation plan only. This document does not represent completed functionality.

## Goal

Replace the current Python-dependent Hugging Face and PEFT LoRA conversion path with a native C++ conversion layer integrated into EasyDifussion-Cpp, while preserving the existing Python llama.cpp converters as a fallback until native coverage is broad enough.

The native path should avoid runtime dependencies on Python, PyTorch, and Transformers.

## Existing integration point

EasyDifussion-Cpp already exposes native model conversion through the `sdkit` binary using:

```text
--convert-model
--convert-output
--convert-type
```

The preferred architecture is to extend this existing native conversion entry point instead of introducing another conversion executable.

The Model Tools UI already manages conversion jobs, temporary outputs, logging, exclusivity, and completion state. That job system can remain largely unchanged. The main change is replacing the Hugging Face branch that currently launches llama.cpp's Python conversion scripts.

## Proposed source layout

```text
source/sdkit3-port-source/
└── src/
    └── conversion/
        ├── hf_converter.h
        ├── hf_converter.cpp
        ├── hf_config.h
        ├── hf_config.cpp
        ├── safetensors_reader.h
        ├── safetensors_reader.cpp
        ├── tensor_mapper.h
        ├── tensor_mapper.cpp
        ├── tokenizer_reader.h
        ├── tokenizer_reader.cpp
        ├── lora_converter.h
        └── lora_converter.cpp
```

## Public conversion interface

```cpp
#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace ed::convert {

enum class OutputType {
    Auto,
    F32,
    F16,
    BF16,
    Q8_0,
};

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    OutputType output_type = OutputType::F16;

    std::function<void(const std::string&)> log;
};

enum class InputKind {
    Unknown,
    HuggingFace,
    Lora,
    Diffusers,
    Checkpoint,
};

InputKind detect_input_kind(const std::filesystem::path& path);

bool convert_huggingface(const Options& options);
bool convert_lora(const Options& options);

}
```

## Input detection

Input detection should be based on repository/file layout rather than Transformers model loading.

```cpp
#include "hf_converter.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace ed::convert {

InputKind detect_input_kind(const fs::path& path) {
    if (fs::is_regular_file(path)) {
        const auto ext = path.extension().string();

        if (ext == ".safetensors" ||
            ext == ".ckpt" ||
            ext == ".pt" ||
            ext == ".pth") {
            return InputKind::Checkpoint;
        }

        return InputKind::Unknown;
    }

    if (!fs::is_directory(path))
        return InputKind::Unknown;

    // PEFT LoRA
    if (fs::exists(path / "adapter_config.json") &&
        (fs::exists(path / "adapter_model.safetensors") ||
         fs::exists(path / "adapter_model.bin"))) {
        return InputKind::Lora;
    }

    // Diffusers
    if (fs::exists(path / "model_index.json"))
        return InputKind::Diffusers;

    // Generic Hugging Face model repository.
    if (fs::exists(path / "config.json"))
        return InputKind::HuggingFace;

    return InputKind::Unknown;
}

}
```

## Hugging Face configuration without Transformers

Treat Hugging Face as a repository/file layout, not as a Transformers runtime dependency.

Parse `config.json` directly and retain only values needed by conversion.

```cpp
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ed::convert {

struct HFConfig {
    std::string model_type;
    std::vector<std::string> architectures;

    uint64_t hidden_size = 0;
    uint64_t intermediate_size = 0;

    uint64_t num_hidden_layers = 0;
    uint64_t num_attention_heads = 0;
    uint64_t num_key_value_heads = 0;

    uint64_t vocab_size = 0;
    uint64_t max_position_embeddings = 0;

    double rope_theta = 10000.0;
    double rms_norm_eps = 1e-5;

    std::string torch_dtype;

    static HFConfig load(
        const std::filesystem::path& model_directory);
};

}
```

Example implementation using `nlohmann::json`:

```cpp
#include "hf_config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

namespace ed::convert {

template<typename T>
static T get_or(
    const json& j,
    const char* key,
    T fallback) {

    if (!j.contains(key) || j[key].is_null())
        return fallback;

    return j[key].get<T>();
}

HFConfig HFConfig::load(
    const std::filesystem::path& directory) {

    std::ifstream stream(directory / "config.json");

    if (!stream)
        throw std::runtime_error(
            "Unable to open Hugging Face config.json");

    json j;
    stream >> j;

    HFConfig cfg;

    cfg.model_type =
        get_or<std::string>(j, "model_type", "");

    cfg.architectures =
        get_or<std::vector<std::string>>(
            j,
            "architectures",
            {});

    cfg.hidden_size =
        get_or<uint64_t>(j, "hidden_size", 0);

    cfg.intermediate_size =
        get_or<uint64_t>(j, "intermediate_size", 0);

    cfg.num_hidden_layers =
        get_or<uint64_t>(j, "num_hidden_layers", 0);

    cfg.num_attention_heads =
        get_or<uint64_t>(j, "num_attention_heads", 0);

    cfg.num_key_value_heads =
        get_or<uint64_t>(
            j,
            "num_key_value_heads",
            cfg.num_attention_heads);

    cfg.vocab_size =
        get_or<uint64_t>(j, "vocab_size", 0);

    cfg.max_position_embeddings =
        get_or<uint64_t>(
            j,
            "max_position_embeddings",
            0);

    cfg.rope_theta =
        get_or<double>(
            j,
            "rope_theta",
            10000.0);

    cfg.rms_norm_eps =
        get_or<double>(
            j,
            "rms_norm_eps",
            1e-5);

    cfg.torch_dtype =
        get_or<std::string>(
            j,
            "torch_dtype",
            "");

    return cfg;
}

}
```

`torch_dtype` is only a field name in the JSON configuration and does not require linking or importing Torch.

## SafeTensors reader

SafeTensors should be the primary native model input format.

The file contains an 8-byte little-endian JSON header length, the JSON header, and then contiguous tensor bytes.

Initial interface:

```cpp
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace ed::convert {

enum class TensorDType {
    Unknown,
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL
};

struct TensorInfo {
    std::string name;
    TensorDType dtype = TensorDType::Unknown;

    std::vector<uint64_t> shape;

    uint64_t begin = 0;
    uint64_t end = 0;
};

class SafeTensorsReader {
public:
    explicit SafeTensorsReader(
        const std::filesystem::path& path);

    const std::unordered_map<
        std::string,
        TensorInfo>& tensors() const {
        return tensors_;
    }

    std::vector<uint8_t> read_tensor(
        const TensorInfo& tensor) const;

private:
    std::filesystem::path path_;
    uint64_t data_offset_ = 0;

    std::unordered_map<
        std::string,
        TensorInfo> tensors_;
};

}
```

Initial implementation:

```cpp
#include "safetensors_reader.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

namespace ed::convert {

static TensorDType parse_dtype(
    const std::string& value) {

    if (value == "F64")  return TensorDType::F64;
    if (value == "F32")  return TensorDType::F32;
    if (value == "F16")  return TensorDType::F16;
    if (value == "BF16") return TensorDType::BF16;
    if (value == "I64")  return TensorDType::I64;
    if (value == "I32")  return TensorDType::I32;
    if (value == "I16")  return TensorDType::I16;
    if (value == "I8")   return TensorDType::I8;
    if (value == "U8")   return TensorDType::U8;
    if (value == "BOOL") return TensorDType::BOOL;

    return TensorDType::Unknown;
}

static uint64_t read_u64_le(std::ifstream& stream) {
    uint8_t raw[8];

    stream.read(
        reinterpret_cast<char*>(raw),
        sizeof(raw));

    if (!stream)
        throw std::runtime_error(
            "Invalid SafeTensors header");

    uint64_t value = 0;

    for (int i = 0; i < 8; ++i)
        value |=
            static_cast<uint64_t>(raw[i])
            << (i * 8);

    return value;
}

SafeTensorsReader::SafeTensorsReader(
    const std::filesystem::path& path)
    : path_(path) {

    std::ifstream stream(
        path,
        std::ios::binary);

    if (!stream)
        throw std::runtime_error(
            "Unable to open SafeTensors file");

    const uint64_t header_size =
        read_u64_le(stream);

    std::string header(
        static_cast<size_t>(header_size),
        '\0');

    stream.read(
        header.data(),
        static_cast<std::streamsize>(
            header_size));

    if (!stream)
        throw std::runtime_error(
            "Truncated SafeTensors header");

    data_offset_ =
        8 + header_size;

    const json root =
        json::parse(header);

    for (auto it = root.begin();
         it != root.end();
         ++it) {

        if (it.key() == "__metadata__")
            continue;

        const auto& entry = it.value();

        TensorInfo tensor;

        tensor.name = it.key();

        tensor.dtype =
            parse_dtype(
                entry.at("dtype")
                    .get<std::string>());

        tensor.shape =
            entry.at("shape")
                .get<std::vector<uint64_t>>();

        const auto offsets =
            entry.at("data_offsets")
                .get<std::vector<uint64_t>>();

        if (offsets.size() != 2)
            throw std::runtime_error(
                "Invalid SafeTensors offsets");

        tensor.begin = offsets[0];
        tensor.end = offsets[1];

        tensors_.emplace(
            tensor.name,
            std::move(tensor));
    }
}

std::vector<uint8_t>
SafeTensorsReader::read_tensor(
    const TensorInfo& tensor) const {

    const uint64_t size =
        tensor.end - tensor.begin;

    std::vector<uint8_t> data(
        static_cast<size_t>(size));

    std::ifstream stream(
        path_,
        std::ios::binary);

    stream.seekg(
        static_cast<std::streamoff>(
            data_offset_ +
            tensor.begin));

    stream.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(size));

    if (!stream)
        throw std::runtime_error(
            "Unable to read tensor " +
            tensor.name);

    return data;
}

}
```

The production implementation should use memory mapping or mapped-file views rather than allocating complete secondary tensor copies.

## Sharded Hugging Face repositories

Native conversion should support repositories such as:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

The index file's `weight_map` identifies the shard containing every tensor.

Suggested representation:

```cpp
struct TensorLocation {
    std::filesystem::path file;
    std::string tensor_name;
};

class HFWeightIndex {
public:
    static HFWeightIndex load(
        const std::filesystem::path& directory);

    const std::unordered_map<
        std::string,
        TensorLocation>& tensors() const;

private:
    std::unordered_map<
        std::string,
        TensorLocation> tensors_;
};
```

## Architecture handlers

Avoid recreating the existing Python converter as one giant C++ source file. Use architecture-specific handlers instead.

```cpp
class ArchitectureHandler {
public:
    virtual ~ArchitectureHandler() = default;

    virtual bool accepts(
        const HFConfig& config) const = 0;

    virtual std::string map_tensor(
        const std::string& hf_name) const = 0;

    virtual void write_metadata(
        GGUFWriter& writer,
        const HFConfig& config) const = 0;
};
```

Example Llama handler:

```cpp
class LlamaHandler final
    : public ArchitectureHandler {

public:
    bool accepts(
        const HFConfig& config) const override {

        return config.model_type == "llama";
    }

    std::string map_tensor(
        const std::string& name) const override {

        if (name == "model.embed_tokens.weight")
            return "token_embd.weight";

        if (name == "model.norm.weight")
            return "output_norm.weight";

        if (name == "lm_head.weight")
            return "output.weight";

        // Layer mappings handled through pattern matching.

        return {};
    }

    void write_metadata(
        GGUFWriter& writer,
        const HFConfig& cfg) const override {

        writer.add_uint32(
            "llama.block_count",
            cfg.num_hidden_layers);

        writer.add_uint32(
            "llama.embedding_length",
            cfg.hidden_size);

        writer.add_uint32(
            "llama.attention.head_count",
            cfg.num_attention_heads);

        writer.add_uint32(
            "llama.attention.head_count_kv",
            cfg.num_key_value_heads);
    }
};
```

Architecture support can then grow incrementally:

```text
LlamaHandler
MistralHandler
Qwen2Handler
Qwen3Handler
GemmaHandler
PhiHandler
DeepSeekHandler
```

## Tokenizer handling

Tokenizer support should also avoid Transformers.

Potential native inputs include:

- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `added_tokens.json`
- SentencePiece `.model` files through the SentencePiece C++ library when required

Tokenizer behavior should be implemented by reproducing the relevant file semantics and defaults, not by embedding a Python runtime.

## PEFT LoRA conversion

LoRA conversion is significantly smaller than full Hugging Face model conversion.

Suggested structures:

```cpp
struct LoraConfig {
    uint32_t rank = 0;
    float alpha = 0.0f;

    std::string base_model;
};

struct LoraPair {
    const TensorInfo* a = nullptr;
    const TensorInfo* b = nullptr;
};
```

Parse `adapter_config.json` directly:

```cpp
LoraConfig load_lora_config(
    const std::filesystem::path& directory) {

    std::ifstream stream(
        directory / "adapter_config.json");

    if (!stream)
        throw std::runtime_error(
            "Missing adapter_config.json");

    nlohmann::json j;
    stream >> j;

    LoraConfig cfg;

    cfg.rank =
        j.value("r", 0u);

    cfg.alpha =
        j.value("lora_alpha",
                static_cast<float>(cfg.rank));

    cfg.base_model =
        j.value(
            "base_model_name_or_path",
            "");

    return cfg;
}
```

Pair A/B tensors from `adapter_model.safetensors`:

```cpp
std::unordered_map<
    std::string,
    LoraPair> pairs;

for (const auto& [name, tensor] :
     reader.tensors()) {

    constexpr std::string_view A =
        ".lora_A.weight";

    constexpr std::string_view B =
        ".lora_B.weight";

    if (name.ends_with(A)) {
        auto base =
            name.substr(
                0,
                name.size() - A.size());

        pairs[base].a = &tensor;
    }
    else if (name.ends_with(B)) {
        auto base =
            name.substr(
                0,
                name.size() - B.size());

        pairs[base].b = &tensor;
    }
}
```

Validate complete pairs before writing:

```cpp
for (const auto& [name, pair] : pairs) {
    if (!pair.a || !pair.b) {
        throw std::runtime_error(
            "Incomplete LoRA pair: " + name);
    }
}
```

A PEFT tensor such as:

```text
base_model.model.model.layers.0.self_attn.q_proj
```

would be mapped to the appropriate GGUF base tensor name, for example:

```text
blk.0.attn_q
```

and written as the corresponding LoRA A/B tensors with adapter metadata.

## Integration with sdkit

The existing conversion command-line interface can remain unchanged.

Conceptually:

```cpp
const auto kind =
    ed::convert::detect_input_kind(
        args.convert_model);

switch (kind) {

case ed::convert::InputKind::HuggingFace:
    return ed::convert::convert_huggingface({
        args.convert_model,
        args.convert_output,
        parse_output_type(args.convert_type),
        [](const std::string& msg) {
            LOG_INFO("%s", msg.c_str());
        }
    }) ? 0 : 1;

case ed::convert::InputKind::Lora:
    return ed::convert::convert_lora({
        args.convert_model,
        args.convert_output,
        parse_output_type(args.convert_type),
        {}
    }) ? 0 : 1;

case ed::convert::InputKind::Diffusers:
case ed::convert::InputKind::Checkpoint:
    return existing_sd_converter(...);

default:
    LOG_ERROR("Unsupported conversion input");
    return 1;
}
```

## UI integration

The current Model Tools backend explicitly probes for a Python 3.13 environment and packages including:

```text
gguf
numpy
requests
sentencepiece
torch
transformers
google.protobuf
```

Once native HF/LoRA conversion is usable, Hugging Face model folders no longer need to be classified as a Python-only conversion source.

The final readiness model can become centered on the native `sdkit` binary instead of maintaining separate native and llama/Python readiness states.

The existing job thread, temporary file, logging, output collision prevention, and completion handling should remain intact.

## Compatibility fallback

Do not remove the existing llama.cpp Python converters immediately.

Recommended transition:

```text
Native C++ supports architecture/input?
        |
       yes
        v
native HF/LoRA -> GGUF

       no
        v
existing llama.cpp Python converter
```

This allows native coverage to grow incrementally while preserving current functionality.

## Version 1 scope

The first useful native milestone should intentionally be narrow:

```text
SafeTensors reader
      v
config.json parser
      v
Llama architecture
      v
tokenizer.json
      v
F16/BF16 GGUF
      v
PEFT LoRA
```

Initial exclusions:

- PyTorch `.bin` model loading
- `.pt`/`.pth` Hugging Face transformer repositories
- remote Hugging Face downloading
- conversion-time quantization beyond straightforward supported output types
- obscure tokenizer implementations
- full llama.cpp architecture parity

The key milestone is not complete architecture coverage. It is proving that a fully native, Transformers-free, Torch-free Hugging Face/PEFT to GGUF pipeline works inside EasyDifussion-Cpp.

## Later work

After the first native path is validated:

1. Add memory-mapped tensor access.
2. Add sharded SafeTensors support.
3. Expand tokenizer compatibility.
4. Add architecture handlers one family at a time.
5. Add parity tests against llama.cpp's Python converter.
6. Compare GGUF metadata and tensor names between native and Python outputs.
7. Run logits/perplexity validation for language models where practical.
8. Remove Python conversion dependencies only after native coverage is considered sufficient.

## Optional oneAPI / SYCL acceleration path

The native converter should be able to use Intel oneAPI as an optional acceleration backend without making oneAPI a hard requirement for conversion. The CPU implementation remains the reference path; oneAPI accelerates tensor-heavy stages when a supported Intel GPU is present.

This is particularly relevant to the H3C XG310 accelerator. H3C documents the XG310 as an Intel SG1 / Gen12 design containing four independent GPU devices. Each GPU has 96 execution units and 8 GB of LPDDR4x, for 32 GB total board memory. The implementation must therefore treat the card as four 8 GB devices rather than assuming one unified 32 GB allocation.

Current Intel oneAPI releases list Intel Server GPU support. Runtime compatibility should still be detected instead of assumed. Startup should enumerate SYCL devices and enable acceleration only when the installed Level Zero/OpenCL stack exposes suitable devices.

### Design rule

oneAPI should sit below the converter as a compute backend:

```text
HF / LoRA parser
      |
      v
Tensor transform plan
      |
      +------------------+
      |                  |
      v                  v
CPU backend        oneAPI/SYCL backend
      |                  |
      +--------+---------+
               |
               v
           GGUF writer
```

Parsing JSON, reading tokenizer metadata, resolving tensor names, and writing GGUF metadata do not belong on the GPU. oneAPI should be used only for operations where parallel execution is useful, such as dtype conversion, transpose/repack operations, quantization preparation, statistics, and optional LoRA tensor math.

### Suggested source layout

```text
source/sdkit3-port-source/src/conversion/
    compute_backend.h
    cpu_compute_backend.cpp
    oneapi_compute_backend.cpp
    oneapi_device_manager.cpp
```

The converter should depend on an abstract interface rather than directly including SYCL code everywhere.

```cpp
class ConversionComputeBackend {
public:
    virtual ~ConversionComputeBackend() = default;

    virtual std::string name() const = 0;

    virtual bool convert_dtype(
        const TensorView& src,
        MutableTensorView& dst,
        TensorDType target) = 0;

    virtual bool transpose_2d(
        const TensorView& src,
        MutableTensorView& dst) = 0;

    virtual bool quantize_prepare(
        const TensorView& src,
        MutableTensorView& dst,
        OutputType type) = 0;
};
```

The CPU implementation is always available. A factory can select oneAPI only when requested and usable:

```cpp
std::unique_ptr<ConversionComputeBackend>
create_compute_backend(bool prefer_oneapi) {
#ifdef ED_ENABLE_ONEAPI
    if (prefer_oneapi) {
        auto backend = create_oneapi_backend();
        if (backend && backend->available())
            return backend;
    }
#endif

    return create_cpu_backend();
}
```

### Device discovery

Use SYCL enumeration rather than device-name assumptions.

```cpp
#include <sycl/sycl.hpp>

std::vector<sycl::device> find_intel_gpus() {
    std::vector<sycl::device> result;

    for (const auto& platform : sycl::platform::get_platforms()) {
        for (const auto& device : platform.get_devices()) {
            if (!device.is_gpu())
                continue;

            const auto vendor =
                device.get_info<sycl::info::device::vendor>();

            if (vendor.find("Intel") != std::string::npos)
                result.push_back(device);
        }
    }

    return result;
}
```

For diagnostics, log at least:

```text
device name
vendor
backend
number of compute units
global memory
maximum allocation size
USM capabilities
```

The XG310 should normally appear as multiple GPU devices. The implementation must not concatenate their memory into a fake 32 GB pointer space.

### Unified Shared Memory

SYCL Unified Shared Memory gives the converter ordinary C/C++ pointer semantics while still allowing device work. Prefer explicit device allocations for larger tensor transforms and shared allocations for smaller control structures when supported.

A simple staging object can look like:

```cpp
struct OneApiBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    sycl::queue* queue = nullptr;

    ~OneApiBuffer() {
        if (ptr && queue)
            sycl::free(ptr, *queue);
    }
};
```

Large model conversion should remain streaming. Do not upload an entire model to an 8 GB XG310 device. Process one tensor or one chunk at a time:

```text
mmap SafeTensors shard
        |
        v
host tensor view
        |
        v
copy chunk -> selected GPU
        |
        v
transform / cast / quantize
        |
        v
copy output chunk -> host
        |
        v
GGUF writer
```

This also makes the same code usable on Intel GPUs with much smaller VRAM.

### Example SYCL dtype conversion

A basic F32-to-F16 conversion kernel can be implemented without Torch or Transformers:

```cpp
void f32_to_f16(
    sycl::queue& q,
    const float* input,
    sycl::half* output,
    size_t count) {

    q.parallel_for(
        sycl::range<1>(count),
        [=](sycl::id<1> i) {
            output[i] =
                static_cast<sycl::half>(input[i]);
        });
}
```

The production backend should submit copies and kernels asynchronously and use SYCL events to avoid unnecessary global queue waits.

### Four-device XG310 scheduling

The XG310 is best used as four conversion workers, each constrained to its own 8 GB memory pool.

```text
Tensor queue
   |
   +--> SG1 GPU 0 -- tensor/chunk A
   +--> SG1 GPU 1 -- tensor/chunk B
   +--> SG1 GPU 2 -- tensor/chunk C
   +--> SG1 GPU 3 -- tensor/chunk D
```

A simple scheduler can maintain one `sycl::queue` per device and assign independent tensors by estimated byte size.

```cpp
struct OneApiWorker {
    sycl::device device;
    sycl::context context;
    sycl::queue queue;
    size_t budget_bytes = 0;
};
```

Do not assume peer-to-peer access between the four SG1 GPUs. The safe baseline is host-mediated staging. If Level Zero later exposes useful peer-copy capability, that can be added as an optional optimization.

For GGUF conversion this topology is still useful because most tensor transforms are independent. Four tensors can be cast/repacked simultaneously and committed to the output in deterministic GGUF order after their associated events complete.

### Deterministic output ordering

Parallel GPU conversion must not make the GGUF file nondeterministic. Separate transformation from file emission:

```text
worker 0 -> transformed tensor 4 --+
worker 1 -> transformed tensor 2 --+
worker 2 -> transformed tensor 3 --+--> ordered completion queue --> GGUF writer
worker 3 -> transformed tensor 1 --+
```

The GGUF writer remains a single ordered writer. GPU workers may finish out of order, but the output tensor ordering and metadata must remain stable.

### LoRA-specific use

LoRA conversion usually does not require expensive inference math, so GPU use is optional. oneAPI becomes more useful when the converter must transpose, cast, normalize, repack, or merge adapters.

If adapter merging is later supported, oneMKL can optionally provide matrix multiplication for operations equivalent to applying the low-rank update, while ordinary GGUF LoRA export can continue to simply map and convert A/B tensors.

Do not require oneDNN or oneMKL for the initial implementation. Raw SYCL kernels are enough for dtype conversion and basic transforms, keeping the dependency surface smaller.

### Build integration

Make oneAPI opt-in at build time:

```cmake
option(ED_ENABLE_ONEAPI
       "Enable oneAPI/SYCL conversion acceleration"
       OFF)

if(ED_ENABLE_ONEAPI)
    target_compile_definitions(sdkit PRIVATE ED_ENABLE_ONEAPI=1)
    target_sources(sdkit PRIVATE
        src/conversion/oneapi_compute_backend.cpp
        src/conversion/oneapi_device_manager.cpp)
endif()
```

The exact compiler/link setup can then be isolated to this option instead of forcing every EasyDifussion-Cpp build to use the DPC++ compiler.

An initial Intel build could use Intel's DPC++/C++ compiler for the translation units containing SYCL code while keeping the backend boundary narrow enough to avoid contaminating unrelated source files.

### Runtime controls

Later command-line options could be added without changing the existing default behavior:

```text
--convert-device cpu
--convert-device oneapi
--convert-device auto
--convert-device oneapi:0
--convert-oneapi-workers 4
```

Recommended default:

```text
--convert-device auto
```

`auto` should prefer the native CPU path until oneAPI device discovery succeeds. If oneAPI initialization or a GPU transform fails, the current tensor can be retried on CPU unless strict GPU mode was explicitly requested.

### Runtime verification

Before depending on the XG310 backend, verify that the installed stack exposes the devices through SYCL/Level Zero. A useful first-run diagnostic should print all detected devices and their memory limits rather than relying on marketing capacity or PCI enumeration alone.

The converter should therefore distinguish:

```text
PCI device present
        !=
Level Zero device usable
        !=
SYCL device usable
        !=
conversion backend validated
```

Only the final state should enable oneAPI acceleration automatically.

### Why this fits the native converter

The native Hugging Face/LoRA converter and oneAPI solve different parts of the same dependency problem:

```text
Python / Transformers / Torch
           |
           X remove
           |
Native C++ parser + GGUF writer
           |
           +--> CPU tensor backend
           |
           +--> oneAPI/SYCL tensor backend
```

This avoids replacing a Python dependency with a vendor runtime dependency at the architecture level. oneAPI is an accelerator backend, not the definition of the converter.

The result should still build and convert models on machines with no Intel GPU at all, while a system containing the XG310 can use its four SG1 devices to parallelize conversion work where that actually provides a benefit.
