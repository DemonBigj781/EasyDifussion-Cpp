#include "load.hpp"

#include <array>
#include <exception>
#include <cstddef>
#include <mutex>
#include <utility>

namespace edcpp::api::load::cpu::translation::cpu { void ensure_registered(); }

namespace edcpp::api::load {
namespace {
constexpr std::size_t backend_slots = static_cast<std::size_t>(Backend::directml) + 1;
constexpr std::size_t resource_slots = static_cast<std::size_t>(ResourceType::count);
struct Translation { TranslationFn invoke = nullptr; RollbackFn rollback = nullptr; std::shared_ptr<const void> module_guard; };
using TranslationTable = std::array<std::array<Translation, resource_slots>, backend_slots>;
std::mutex& translation_mutex() {
    static std::mutex mutex;
    return mutex;
}
TranslationTable& translations() {
    static TranslationTable table{};
    return table;
}
}

bool register_translation(Backend backend, ResourceType resource_type,
                          TranslationFn translation, RollbackFn rollback, std::shared_ptr<const void> module_guard) noexcept {
    const auto backend_index = static_cast<std::size_t>(backend);
    const auto resource_index = static_cast<std::size_t>(resource_type);
    if (backend == Backend::none || backend_index >= backend_slots ||
        resource_index >= resource_slots || translation == nullptr || rollback == nullptr) return false;
    // A route is immutable after first registration: changing an unload
    // implementation while resources are alive could release with the wrong owner.
    std::lock_guard<std::mutex> lock(translation_mutex());
    auto& slot = translations()[backend_index][resource_index];
    if (slot.invoke != nullptr) return false;
    slot = {translation, rollback, std::move(module_guard)};
    return true;
}

Result load_resource(Backend backend, const Request& request) {
    // Explicit symbol reference retains CPU translation objects in static archives.
    if (backend == Backend::cpu) cpu::translation::cpu::ensure_registered();
    if (request.data == nullptr || request.size == 0 || request.size > request.max_bytes) {
        Result rejected;
        rejected.diagnostic = "load requires non-empty bytes within the requested allocation budget";
        return rejected;
    }
    const auto backend_index = static_cast<std::size_t>(backend);
    const auto resource_index = static_cast<std::size_t>(request.resource_type);
    Translation selected;
    if (backend != Backend::none && backend_index < backend_slots && resource_index < resource_slots) {
        std::lock_guard<std::mutex> lock(translation_mutex());
        selected = translations()[backend_index][resource_index];
    }
    if (selected.invoke == nullptr) {
        Result result;
        result.diagnostic = "Common load has no registered translation for the requested backend and resource type";
        return result;
    }
    // Do not hold a registry lock while invoking backend code.
    Result result;
    try {
        selected.invoke(request, result);
    } catch (const std::exception& error) {
        if (result.resource.native_handle != nullptr) selected.rollback(result.resource);
        result = {};
        result.diagnostic = std::string("load translation failed: ") + error.what();
        return result;
    } catch (...) {
        if (result.resource.native_handle != nullptr) selected.rollback(result.resource);
        result = {};
        result.diagnostic = "load translation failed with an unknown exception";
        return result;
    }
    const bool valid = result.loaded && result.resource.loaded() &&
        result.resource.backend == backend &&
        result.resource.resource_type == request.resource_type &&
        result.resource.size == request.size &&
        result.resource.size <= request.max_bytes &&
        result.resource.device_index == (backend == Backend::cpu ? 0 : request.device_index);
    if (!valid) {
        // The originating translation owns allocator knowledge, even when its
        // reported metadata is malformed. Never guess an allocator here.
        if (result.resource.native_handle != nullptr) selected.rollback(result.resource);
        result.loaded = false;
        result.resource = {};
        if (result.diagnostic.empty()) result.diagnostic = "load translation returned an invalid resource identity or size";
    }
    return result;
}

bool register_model_translation(Backend backend, ModelTranslationFn translation, RollbackFn rollback, std::shared_ptr<const void> module_guard) noexcept {
    return register_translation(backend, ResourceType::model, translation, rollback, std::move(module_guard));
}

Result load_model(Backend backend, const Request& request) {
    Request model_request = request;
    model_request.resource_type = ResourceType::model;
    return load_resource(backend, model_request);
}

} // namespace edcpp::api::load
