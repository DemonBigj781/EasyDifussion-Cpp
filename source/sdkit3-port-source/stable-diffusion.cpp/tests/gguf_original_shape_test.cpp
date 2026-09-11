#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "model_io/gguf_io.h"

namespace {

std::filesystem::path temporary_gguf_path() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("sdkit-gguf-original-shape-" + std::to_string(nonce) + ".gguf");
}

bool write_fixture(const std::filesystem::path& path) {
    ggml_init_params params;
    params.mem_size   = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    ggml_context* tensor_context = ggml_init(params);
    gguf_context* gguf_context   = gguf_init_empty();
    if (tensor_context == nullptr || gguf_context == nullptr) {
        if (gguf_context != nullptr) {
            gguf_free(gguf_context);
        }
        if (tensor_context != nullptr) {
            ggml_free(tensor_context);
        }
        return false;
    }

    ggml_tensor* tensor = ggml_new_tensor_2d(tensor_context, GGML_TYPE_Q4_K, 256, 45);
    ggml_set_name(tensor, "conv_in.weight");
    std::memset(tensor->data, 0, ggml_nbytes(tensor));
    const int64_t original_shape[] = {320, 4, 3, 3};
    gguf_set_arr_data(
        gguf_context,
        "comfy.gguf.orig_shape.conv_in.weight",
        GGUF_TYPE_INT64,
        original_shape,
        4);
    gguf_add_tensor(gguf_context, tensor);
    const bool written = gguf_write_to_file(gguf_context, path.string().c_str(), false);
    gguf_free(gguf_context);
    ggml_free(tensor_context);
    return written;
}

}  // namespace

int main() {
    const std::filesystem::path path = temporary_gguf_path();
    if (!write_fixture(path)) {
        std::cerr << "failed to write GGUF fixture\n";
        return 1;
    }

    std::vector<TensorStorage> tensors;
    std::string error;
    const bool loaded = read_gguf_file(path.string(), tensors, &error);
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);
    if (!loaded) {
        std::cerr << "failed to read GGUF fixture: " << error << '\n';
        return 1;
    }
    if (tensors.size() != 1) {
        std::cerr << "expected one tensor, got " << tensors.size() << '\n';
        return 1;
    }

    const TensorStorage& tensor = tensors.front();
    const bool logical_shape_ok = tensor.n_dims == 4 && tensor.ne[0] == 3 && tensor.ne[1] == 3 &&
                                  tensor.ne[2] == 4 && tensor.ne[3] == 320;
    const bool storage_shape_ok = tensor.storage_n_dims == 2 && tensor.storage_ne[0] == 256 &&
                                  tensor.storage_ne[1] == 45;
    if (!logical_shape_ok || !storage_shape_ok || tensor.nelements() != tensor.storage_nelements()) {
        std::cerr << "GGUF original shape was not restored correctly: " << tensor.to_string() << '\n';
        return 1;
    }
    return 0;
}
