#pragma once

#include "common/api.hpp"

#include <cstdint>
#include <string>

namespace edcpp::api::load {

enum class ResourceType : std::uint8_t {
    model = 0,
    vae,
    unet,
    clip,
    clip_vision,
    condition,
    image,
    latent,
    mask,
    count,
};

struct Request {
    const void* data = nullptr;
    std::uint64_t size = 0;
    int device_index = 0;
    ResourceType resource_type = ResourceType::model;
};

struct Resource {
    Backend backend = Backend::none;
    void* native_handle = nullptr;
    std::uint64_t size = 0;
    int device_index = -1;
    ResourceType resource_type = ResourceType::model;
    bool loaded() const noexcept { return backend != Backend::none && native_handle != nullptr && size != 0; }
};

struct Result {
    bool loaded = false;
    Resource resource;
    std::string diagnostic;
};

using TranslationFn = Result (*)(const Request&);
bool register_translation(Backend backend, ResourceType resource_type,
                          TranslationFn translation) noexcept;
Result load_resource(Backend backend, const Request& request);

using ModelTranslationFn = TranslationFn;
bool register_model_translation(Backend backend, ModelTranslationFn translation) noexcept;
Result load_model(Backend backend, const Request& request);
Result normalize(Result result);

} // namespace edcpp::api::load
