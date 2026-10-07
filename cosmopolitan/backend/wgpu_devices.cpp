/* SPDX-License-Identifier: MIT */
#include "cosmo-webgpu.h"
#include "vulkan_loader.h"
#include "register.h"
#include <webgpu.h>
#include <wgpu.h>
#include <cosmo.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

namespace {
std::mutex metadata_mutex, creation_mutex;
std::string requested_provider = "auto", requested_device = "auto";
bool instances_started = false;
std::deque<std::string> errors;
struct DeviceRecord {
    std::string selector, provider, name, stable_id;
    cosmo_webgpu_device_info info{};
    DeviceRecord(const char *s, const char *p, const char *n, int software,
                 uint32_t type, const char *id)
        : selector(s), provider(p), name(n), stable_id(id ? id : "") {
        info.selector = selector.c_str(); info.provider = provider.c_str();
        info.name = name.c_str(); info.stable_id = stable_id.c_str();
        info.software = software; info.adapter_type = type;
    }
};
struct UnavailableRecord {
    std::string provider, name, reason;
    cosmo_webgpu_unavailable_info info{};
    UnavailableRecord(const char *p, const char *n, uint32_t type, const char *r)
        : provider(p), name(n), reason(r) {
        info = {provider.c_str(), name.c_str(), type, reason.c_str()};
    }
};
std::deque<std::unique_ptr<DeviceRecord>> devices;
std::deque<std::unique_ptr<UnavailableRecord>> unavailable;
const DeviceRecord *selected = nullptr;
int fail_locked(std::string message) {
    errors.push_back(std::move(message));
    return -1;
}
bool hardware(const DeviceRecord &d) {
    return !d.info.software && (d.info.adapter_type == WGPUAdapterType_DiscreteGPU ||
                               d.info.adapter_type == WGPUAdapterType_IntegratedGPU);
}
bool permitted(const DeviceRecord &d) {
    return requested_provider == "auto" || requested_provider == d.provider;
}
const DeviceRecord *resolve_locked(const char *selector) {
    std::string wanted = selector && *selector ? selector : "auto";
    if (wanted == "auto") wanted = requested_device;
    if (wanted != "auto") {
        const DeviceRecord *match = nullptr;
        for (const auto &d : devices) {
            if (!permitted(*d)) continue;
            if (wanted != d->selector && (d->stable_id.empty() || wanted != d->stable_id)) continue;
            if (match) return nullptr;
            match = d.get();
        }
        return match;
    }
    for (const auto &d : devices) if (permitted(*d) && hardware(*d)) return d.get();
    /* Native automatic selection requires hardware. An explicitly named
     * native software adapter remains selectable for real loader verification. */
    if (requested_provider == "native") return nullptr;
    for (const auto &d : devices)
        if (permitted(*d) && d->provider == "embedded" && d->info.software) return d.get();
    return nullptr;
}
const DeviceRecord *current_locked() { return selected ? selected : resolve_locked("auto"); }
}

extern "C" int cosmo_webgpu_configure(const char *provider, const char *device) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    if (!provider || (strcmp(provider, "auto") && strcmp(provider, "native") &&
                      strcmp(provider, "embedded")))
        return fail_locked("WebGPU provider must be auto, native, or embedded");
    if (!device || !*device || strlen(device) > 255)
        return fail_locked("WebGPU device must be auto or an available selector");
    if (instances_started && (requested_provider != provider || requested_device != device))
        return fail_locked("WebGPU provider/device policy is initialized; restart to change it");
    requested_provider = provider; requested_device = device;
    return 0;
}
extern "C" int cosmo_webgpu_initialize(void) {
    static std::once_flag once;
    static int status = -1;
    std::call_once(once, [] { status = cosmo_wgpu_lavapipe_register(); });
    if (status) {
        std::lock_guard<std::mutex> guard(metadata_mutex);
        fail_locked("The linked software Vulkan provider could not be registered");
    }
    return status;
}
extern "C" const char *cosmo_webgpu_requested_provider(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    return requested_provider.c_str();
}
extern "C" const char *cosmo_webgpu_requested_device(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    return requested_device.c_str();
}
extern "C" const char *cosmo_webgpu_last_error(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    return errors.empty() ? "" : errors.back().c_str();
}
extern "C" void *cosmo_webgpu_create_instance(const char *provider, const void *descriptor) {
    if (!provider || (strcmp(provider, "native") && strcmp(provider, "embedded"))) return nullptr;
    if (IsLinux() && !strcmp(provider, "native") && gettid() != getpid()) {
        std::lock_guard<std::mutex> guard(metadata_mutex);
        fail_locked("Native Vulkan must initialize on the original main thread on Linux");
        return nullptr;
    }
    if (cosmo_webgpu_initialize()) return nullptr;
    std::lock_guard<std::mutex> creation_guard(creation_mutex);
    {
        std::lock_guard<std::mutex> guard(metadata_mutex);
        if (requested_provider != "auto" && requested_provider != provider) {
            fail_locked("Instance provider differs from the configured WebGPU policy");
            return nullptr;
        }
        instances_started = true;
    }
    if (cosmo_wgpu_vulkan_select(provider)) return nullptr;
    WGPUInstanceDescriptor native = descriptor
        ? *static_cast<const WGPUInstanceDescriptor *>(descriptor)
        : WGPU_INSTANCE_DESCRIPTOR_INIT;
    WGPUInstanceExtras extras{};
    extras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
    extras.chain.next = native.nextInChain;
    extras.backends = WGPUInstanceBackend_Vulkan;
    /* Native callbacks require a reverse ABI/TLS bridge. Do not enable them
     * through debug/validation flags or inherited WGPU_* environment values. */
    extras.flags = 0;
    native.nextInChain = &extras.chain;
    WGPUInstance instance = wgpuCreateInstance(&native);
    if (!instance) {
        std::lock_guard<std::mutex> guard(metadata_mutex);
        fail_locked(std::string("Could not create the ") + provider + " Vulkan WebGPU instance");
    }
    return instance;
}
extern "C" void cosmo_webgpu_register_device_metadata(const char *selector, const char *provider,
    const char *name, int software, uint32_t type, const char *stable_id) {
    if (!selector || !*selector || !provider || !name) return;
    std::lock_guard<std::mutex> guard(metadata_mutex);
    for (const auto &d : devices) if (d->selector == selector) return;
    devices.emplace_back(new DeviceRecord(selector, provider, name, software, type, stable_id));
}
extern "C" const cosmo_webgpu_device_info *cosmo_webgpu_device_metadata(const char *selector) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    if (!selector || !strcmp(selector, "auto")) {
        const auto *d = current_locked();
        return d ? &d->info : nullptr;
    }
    const DeviceRecord *match = nullptr;
    for (const auto &d : devices) {
        if (d->selector != selector && (d->stable_id.empty() || d->stable_id != selector)) continue;
        if (match) return nullptr;
        match = d.get();
    }
    return match ? &match->info : nullptr;
}
extern "C" size_t cosmo_webgpu_device_count(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex); return devices.size();
}
extern "C" const cosmo_webgpu_device_info *cosmo_webgpu_device_at(size_t index) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    return index < devices.size() ? &devices[index]->info : nullptr;
}
extern "C" int cosmo_webgpu_select_device(const char *selector) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    const auto *d = resolve_locked(selector);
    if (!d) {
        std::string message = "No compatible WebGPU device matches ";
        message += selector && *selector ? selector : "auto";
        message += " (provider=" + requested_provider + ")";
        if (requested_provider == "native" && (!selector || !strcmp(selector, "auto")) && requested_device == "auto")
            message += "; automatic native selection requires a hardware GPU. Check the installed Vulkan driver and ShaderF16 support";
        return fail_locked(std::move(message));
    }
    selected = d; return 0;
}
extern "C" const char *cosmo_webgpu_selected_device(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    const auto *d = current_locked(); return d ? d->selector.c_str() : "";
}
extern "C" const char *cosmo_webgpu_default_device(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    const auto *d = resolve_locked("auto"); return d ? d->selector.c_str() : "";
}
extern "C" const char *cosmo_webgpu_adapter_name(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    const auto *d = current_locked(); return d ? d->name.c_str() : "";
}
extern "C" int cosmo_webgpu_adapter_is_software(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    const auto *d = current_locked(); return d ? d->info.software : -1;
}
extern "C" const char *cosmo_webgpu_provider_name(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    const auto *d = current_locked(); return d ? d->provider.c_str() : "unselected";
}
extern "C" void cosmo_webgpu_note_adapter(const char *, int) {
    /* Retained ABI. Enumeration never changes the selected device. */
}
extern "C" void cosmo_webgpu_note_unavailable_adapter(const char *provider, const char *name,
    uint32_t type, const char *reason) {
    if (!provider || !name || !reason) return;
    std::lock_guard<std::mutex> guard(metadata_mutex);
    unavailable.emplace_back(new UnavailableRecord(provider, name, type, reason));
    fprintf(stderr, "WebGPU adapter unavailable: provider=%s name=%s reason=%s\n", provider, name, reason);
}
extern "C" size_t cosmo_webgpu_unavailable_count(void) {
    std::lock_guard<std::mutex> guard(metadata_mutex); return unavailable.size();
}
extern "C" const cosmo_webgpu_unavailable_info *cosmo_webgpu_unavailable_at(size_t index) {
    std::lock_guard<std::mutex> guard(metadata_mutex);
    return index < unavailable.size() ? &unavailable[index]->info : nullptr;
}
