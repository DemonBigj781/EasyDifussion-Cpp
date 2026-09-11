#include "gguf_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "core/util.h"
#include "gguf.h"
#include "gguf_reader_ext.h"

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

static bool checked_nelements(const int64_t* dimensions,
                              size_t dimension_count,
                              int64_t& result) {
    int64_t count = 1;
    for (size_t i = 0; i < dimension_count; i++) {
        const int64_t dimension = dimensions[i];
        if (dimension <= 0 || count > std::numeric_limits<int64_t>::max() / dimension) {
            return false;
        }
        count *= dimension;
    }
    result = count;
    return true;
}

static bool restore_comfy_original_shape(const gguf_context* context,
                                         const std::string& tensor_name,
                                         TensorStorage& tensor_storage,
                                         std::string* error) {
    const std::string key = "comfy.gguf.orig_shape." + tensor_name;
    const int64_t key_id  = gguf_find_key(context, key.c_str());
    if (key_id < 0) {
        return true;
    }
    if (gguf_get_kv_type(context, key_id) != GGUF_TYPE_ARRAY) {
        set_error(error, "invalid original-shape metadata for tensor '" + tensor_name + "'");
        return false;
    }

    const size_t dimension_count = gguf_get_arr_n(context, key_id);
    if (dimension_count == 0 || dimension_count > SD_MAX_DIMS) {
        set_error(error, "invalid original-shape dimensions for tensor '" + tensor_name + "'");
        return false;
    }

    const void* data = gguf_get_arr_data(context, key_id);
    if (data == nullptr) {
        set_error(error, "missing original-shape data for tensor '" + tensor_name + "'");
        return false;
    }

    std::vector<int64_t> original_shape(dimension_count);
    switch (gguf_get_arr_type(context, key_id)) {
        case GGUF_TYPE_INT32: {
            const auto* dimensions = static_cast<const int32_t*>(data);
            for (size_t i = 0; i < dimension_count; i++) {
                original_shape[i] = dimensions[i];
            }
            break;
        }
        case GGUF_TYPE_UINT32: {
            const auto* dimensions = static_cast<const uint32_t*>(data);
            for (size_t i = 0; i < dimension_count; i++) {
                original_shape[i] = dimensions[i];
            }
            break;
        }
        case GGUF_TYPE_INT64: {
            const auto* dimensions = static_cast<const int64_t*>(data);
            for (size_t i = 0; i < dimension_count; i++) {
                original_shape[i] = dimensions[i];
            }
            break;
        }
        case GGUF_TYPE_UINT64: {
            const auto* dimensions = static_cast<const uint64_t*>(data);
            for (size_t i = 0; i < dimension_count; i++) {
                if (dimensions[i] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    set_error(error, "original-shape dimension is too large for tensor '" + tensor_name + "'");
                    return false;
                }
                original_shape[i] = static_cast<int64_t>(dimensions[i]);
            }
            break;
        }
        default:
            set_error(error, "unsupported original-shape metadata type for tensor '" + tensor_name + "'");
            return false;
    }

    int64_t storage_elements = 0;
    int64_t logical_elements = 0;
    if (!checked_nelements(tensor_storage.ne, tensor_storage.n_dims, storage_elements) ||
        !checked_nelements(original_shape.data(), original_shape.size(), logical_elements)) {
        set_error(error, "invalid or overflowing original-shape dimensions for tensor '" + tensor_name + "'");
        return false;
    }
    if (storage_elements != logical_elements) {
        set_error(error, "original-shape element count mismatch for tensor '" + tensor_name + "'");
        return false;
    }

    tensor_storage.storage_n_dims = tensor_storage.n_dims;
    for (int i = 0; i < SD_MAX_DIMS; i++) {
        tensor_storage.storage_ne[i] = tensor_storage.ne[i];
        tensor_storage.ne[i]         = 1;
    }
    tensor_storage.n_dims = static_cast<int>(dimension_count);
    for (size_t i = 0; i < dimension_count; i++) {
        tensor_storage.ne[i] = original_shape[dimension_count - i - 1];
    }
    return true;
}

bool is_gguf_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4];

    file.read(magic, sizeof(magic));
    if (!file) {
        return false;
    }
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) {
            return false;
        }
    }

    return true;
}

bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error) {
    tensor_storages.clear();

    gguf_context* ctx_gguf_ = nullptr;
    ggml_context* ctx_meta_ = nullptr;

    ctx_gguf_ = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    if (!ctx_gguf_) {
        GGUFReader gguf_reader;
        if (!gguf_reader.load(file_path)) {
            set_error(error, "failed to open '" + file_path + "' with GGUFReader");
            return false;
        }

        size_t data_offset = gguf_reader.data_offset();
        for (const auto& gguf_tensor_info : gguf_reader.tensors()) {
            TensorStorage tensor_storage(
                gguf_tensor_info.name,
                gguf_tensor_info.type,
                gguf_tensor_info.shape.data(),
                static_cast<int>(gguf_tensor_info.shape.size()),
                0,
                data_offset + gguf_tensor_info.offset);

            tensor_storages.push_back(tensor_storage);
        }

        return true;
    }

    int n_tensors = static_cast<int>(gguf_get_n_tensors(ctx_gguf_));

    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name   = gguf_get_tensor_name(ctx_gguf_, i);
        ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset      = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        TensorStorage tensor_storage(name, dummy->type, dummy->ne, ggml_n_dims(dummy), 0, offset);

        if (!restore_comfy_original_shape(ctx_gguf_, name, tensor_storage, error)) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            return false;
        }

        if (ggml_nbytes(dummy) != tensor_storage.nbytes()) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            set_error(error, "size mismatch for tensor '" + name + "'");
            return false;
        }

        tensor_storages.push_back(tensor_storage);
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error) {
    gguf_context* gguf_ctx = gguf_init_empty();
    if (gguf_ctx == nullptr) {
        set_error(error, "gguf_init_empty failed");
        return false;
    }

    for (const TensorWriteInfo& write_tensor : tensors) {
        ggml_tensor* tensor = write_tensor.tensor;
        if (tensor == nullptr) {
            set_error(error, "null tensor cannot be written to GGUF");
            gguf_free(gguf_ctx);
            return false;
        }
        gguf_add_tensor(gguf_ctx, tensor);
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    bool success = gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    if (!success) {
        set_error(error, "failed to write GGUF file '" + file_path + "'");
    }
    gguf_free(gguf_ctx);
    return success;
}

GGUFStreamingWriter::~GGUFStreamingWriter() {
    close();
}

bool GGUFStreamingWriter::write_metadata(const std::string& file_path,
                                         const std::vector<TensorWritePlan>& tensors,
                                         std::string* error) {
    close();
    tensors_   = tensors;
    file_size_ = 0;

    size_t meta_mem = 1 * 1024 * 1024 + tensors.size() * ggml_tensor_overhead();
    meta_ctx_       = ggml_init({meta_mem, nullptr, true});
    if (meta_ctx_ == nullptr) {
        set_error(error, "ggml_init failed for GGUF metadata");
        return false;
    }

    gguf_ctx_ = gguf_init_empty();
    if (gguf_ctx_ == nullptr) {
        set_error(error, "gguf_init_empty failed");
        close();
        return false;
    }

    for (const TensorWritePlan& plan : tensors) {
        ggml_tensor* tensor = ggml_new_tensor(meta_ctx_, plan.type, plan.n_dims, plan.ne);
        if (tensor == nullptr) {
            set_error(error, "ggml_new_tensor failed for tensor '" + plan.name + "'");
            close();
            return false;
        }
        ggml_set_name(tensor, plan.name.c_str());
        gguf_add_tensor(gguf_ctx_, tensor);
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    FILE* file = fopen(file_path.c_str(), "wb+");
    if (file == nullptr) {
        set_error(error, "failed to open output file '" + file_path + "'");
        close();
        return false;
    }

    // ggml exposes GGUF metadata writing through FILE* only. Keep FILE usage
    // isolated here; tensor data is written through std::fstream by the shared
    // streaming pipeline.
    if (!gguf_write_to_file_ptr(gguf_ctx_, file, true)) {
        fclose(file);
        set_error(error, "failed to write GGUF metadata to '" + file_path + "'");
        close();
        return false;
    }
    fclose(file);

    const uint64_t data_start = gguf_get_meta_size(gguf_ctx_);
    tensor_offsets_.resize(tensors.size());
    file_size_ = data_start;
    for (size_t i = 0; i < tensors.size(); i++) {
        tensor_offsets_[i] = data_start + gguf_get_tensor_offset(gguf_ctx_, static_cast<int64_t>(i));
        file_size_         = std::max(file_size_, tensor_offsets_[i] + tensors[i].nbytes());
    }
    return true;
}

bool GGUFStreamingWriter::write_tensor(std::ostream& output,
                                       size_t tensor_index,
                                       const uint8_t* data,
                                       size_t size,
                                       std::string* error) const {
    if (tensor_index >= tensors_.size() || tensor_index >= tensor_offsets_.size()) {
        set_error(error, "invalid GGUF tensor index");
        return false;
    }
    const TensorWritePlan& plan = tensors_[tensor_index];
    if (size != plan.nbytes()) {
        set_error(error, "size mismatch while writing tensor '" + plan.name + "'");
        return false;
    }
    output.seekp(static_cast<std::streamoff>(tensor_offsets_[tensor_index]), std::ios::beg);
    if (!output) {
        set_error(error, "failed to seek output for tensor '" + plan.name + "'");
        return false;
    }
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    if (!output) {
        set_error(error, "failed to write tensor '" + plan.name + "'");
        return false;
    }
    return true;
}

uint64_t GGUFStreamingWriter::file_size() const {
    return file_size_;
}

void GGUFStreamingWriter::close() {
    tensor_offsets_.clear();
    tensors_.clear();
    file_size_ = 0;
    if (gguf_ctx_ != nullptr) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
    }
    if (meta_ctx_ != nullptr) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
}
