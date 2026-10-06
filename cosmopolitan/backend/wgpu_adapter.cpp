/* SPDX-License-Identifier: MIT */
#include "cosmo-webgpu.h"
#include "vulkan_loader.h"
#include "register.h"
#include <atomic>
#include <mutex>
#include <string>

static std::atomic<uint64_t> graphs{0}, submissions{0}, dispatches{0}, readbacks{0}, matmuls{0};
static std::string adapter_name;
static int adapter_software = -1;
static std::mutex adapter_mutex;

extern "C" int cosmo_webgpu_initialize(void) {
    static std::once_flag once;
    static int status = -1;
    std::call_once(once, [] {
        /* Select first: failed registration must also prevent native fallback. */
        cosmo_wgpu_vulkan_select("embedded");
        if (cosmo_wgpu_lavapipe_register() == 0) {
            status = cosmo_wgpu_vulkan_select("embedded");
        }
    });
    return status;
}
extern "C" const char *cosmo_webgpu_adapter_name(void) {
    std::lock_guard<std::mutex> guard(adapter_mutex);
    return adapter_name.c_str();
}
extern "C" int cosmo_webgpu_adapter_is_software(void) {
    std::lock_guard<std::mutex> guard(adapter_mutex);
    return adapter_software;
}
extern "C" void cosmo_webgpu_note_adapter(const char *name, int software) {
    std::lock_guard<std::mutex> guard(adapter_mutex);
    /* The backend has one process-lifetime instance and one selected adapter. */
    if (adapter_name.empty()) {
        adapter_name = name;
        adapter_software = software;
    }
}
extern "C" uint64_t cosmo_webgpu_graph_count(void) { return graphs.load(); }
extern "C" uint64_t cosmo_webgpu_submission_count(void) { return submissions.load(); }
extern "C" uint64_t cosmo_webgpu_dispatch_count(void) { return dispatches.load(); }
extern "C" uint64_t cosmo_webgpu_readback_count(void) { return readbacks.load(); }
extern "C" void cosmo_webgpu_note_graph(void) { ++graphs; }
extern "C" void cosmo_webgpu_note_submission(void) { ++submissions; }
extern "C" void cosmo_webgpu_note_dispatch(void) { ++dispatches; }
extern "C" void cosmo_webgpu_note_readback(void) { ++readbacks; }

extern "C" uint64_t cosmo_webgpu_matmul_dispatch_count(void) { return matmuls.load(); }
extern "C" void cosmo_webgpu_note_matmul_dispatch(void) { ++matmuls; }
extern "C" const char *cosmo_webgpu_provider_name(void) { return cosmo_wgpu_vulkan_selected_provider(); }
extern "C" unsigned long cosmo_webgpu_native_loader_open_count(void) { return cosmo_wgpu_vulkan_native_open_count(); }
