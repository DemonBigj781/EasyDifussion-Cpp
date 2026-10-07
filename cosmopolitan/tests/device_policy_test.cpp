/* Policy unit test: real wgpu_devices.cpp, explicitly mocked provider factory.
 * No Vulkan loader, physical GPU, Mesa execution, or WGPU device is exercised.
 * Run each scenario in a fresh process because production policy is global.
 */
#include "cosmo-webgpu.h"
#include "register.h"
#include "vulkan_loader.h"
#include <webgpu.h>
#include <wgpu.h>
#include <cosmo.h>
#include <pthread.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

static int registration_calls, selection_calls, factory_calls;
static std::string factory_provider;
static const WGPUChainedStruct *expected_tail;
static int instance_token;
static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
extern "C" int cosmo_wgpu_lavapipe_register(void) { ++registration_calls; return 0; }
extern "C" int cosmo_wgpu_vulkan_select(const char *provider) {
    ++selection_calls; factory_provider = provider; return 0;
}
extern "C" WGPUInstance wgpuCreateInstance(const WGPUInstanceDescriptor *descriptor) {
    ++factory_calls;
    check(descriptor && descriptor->nextInChain, "factory requires extras");
    const auto *extras = reinterpret_cast<const WGPUInstanceExtras *>(descriptor->nextInChain);
    check(extras->chain.sType == static_cast<WGPUSType>(WGPUSType_InstanceExtras), "wrong extras type");
    check(extras->backends == WGPUInstanceBackend_Vulkan, "factory must select Vulkan");
    check(extras->flags == 0, "native debug callbacks must remain disabled");
    check(extras->chain.next == expected_tail, "caller descriptor chain was lost");
    return reinterpret_cast<WGPUInstance>(&instance_token);
}
static void record(const char *selector, const char *provider, bool software,
                   WGPUAdapterType type, const char *id = "") {
    cosmo_webgpu_register_device_metadata(selector, provider, selector, software, type, id);
}
static void chosen(const char *expected) {
    check(!strcmp(cosmo_webgpu_selected_device(), expected), "unexpected selected device");
}
static void *native_from_thread(void *result) {
    *static_cast<void **>(result) = cosmo_webgpu_create_instance("native", nullptr);
    return nullptr;
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        const std::string scenario(argv[1]);
        if (scenario == "hardware-first") {
            record("WebGPU0", "embedded", true, WGPUAdapterType_CPU);
            record("WebGPU1", "native", true, WGPUAdapterType_CPU);
            record("WebGPU2", "native", false, WGPUAdapterType_Unknown);
            record("WebGPU3", "native", false, WGPUAdapterType_IntegratedGPU, "uuid:hardware");
            check(!cosmo_webgpu_select_device("auto"), "automatic hardware selection failed"); chosen("WebGPU3");
            check(!cosmo_webgpu_adapter_is_software(), "physical adapter mislabeled software");
            check(!cosmo_webgpu_device_metadata("uuid:hardware")->memory_known, "unknown memory must stay unknown");
        } else if (scenario == "embedded-fallback") {
            record("WebGPU0", "native", true, WGPUAdapterType_CPU);
            record("WebGPU1", "native", false, WGPUAdapterType_Unknown);
            record("WebGPU2", "embedded", true, WGPUAdapterType_CPU);
            check(!cosmo_webgpu_select_device("auto"), "embedded fallback failed"); chosen("WebGPU2");
            check(cosmo_webgpu_adapter_is_software() == 1, "embedded software mislabeled");
        } else if (scenario == "native-requires-hardware") {
            check(!cosmo_webgpu_configure("native", "auto"), "configure failed");
            record("WebGPU0", "native", true, WGPUAdapterType_CPU);
            record("WebGPU1", "native", false, WGPUAdapterType_Unknown);
            record("WebGPU2", "embedded", true, WGPUAdapterType_CPU);
            check(cosmo_webgpu_select_device("auto") != 0, "native auto accepted software or Unknown");
            check(strstr(cosmo_webgpu_last_error(), "requires a hardware GPU"), "missing hardware diagnostic");
            check(!*cosmo_webgpu_default_device(), "native auto silently fell back");
        } else if (scenario == "explicit-native-software") {
            check(!cosmo_webgpu_configure("native", "WebGPU0"), "configure failed");
            record("WebGPU0", "native", true, WGPUAdapterType_CPU);
            check(!cosmo_webgpu_select_device("auto"), "explicit native software test adapter rejected"); chosen("WebGPU0");
            check(cosmo_webgpu_adapter_is_software() == 1, "native software mislabeled hardware");
        } else if (scenario == "missing-and-ambiguous") {
            check(cosmo_webgpu_select_device("missing") != 0, "missing selector accepted");
            record("WebGPU0", "native", false, WGPUAdapterType_DiscreteGPU, "uuid:duplicate");
            record("WebGPU1", "native", false, WGPUAdapterType_DiscreteGPU, "uuid:duplicate");
            check(cosmo_webgpu_select_device("uuid:duplicate") != 0, "ambiguous stable ID accepted");
            check(!cosmo_webgpu_device_metadata("uuid:duplicate"), "ambiguous metadata accepted");
            check(!cosmo_webgpu_configure("embedded", "WebGPU0"), "configure failed");
            check(cosmo_webgpu_select_device("auto") != 0, "cross-provider selector accepted");
            check(cosmo_webgpu_configure("invalid", "auto") != 0, "invalid provider accepted");
        } else if (scenario == "configured-default") {
            check(!cosmo_webgpu_configure("auto", "uuid:second"), "configure failed");
            record("WebGPU0", "native", false, WGPUAdapterType_DiscreteGPU, "uuid:first");
            record("WebGPU1", "native", false, WGPUAdapterType_IntegratedGPU, "uuid:second");
            check(!strcmp(cosmo_webgpu_default_device(), "WebGPU1"), "saved default not resolved");
            check(!cosmo_webgpu_select_device("WebGPU0"), "explicit request failed"); chosen("WebGPU0");
            check(!strcmp(cosmo_webgpu_default_device(), "WebGPU1"), "last request changed saved default");
            check(!cosmo_webgpu_select_device("auto"), "default request failed"); chosen("WebGPU1");
        } else if (scenario == "immutable-factory") {
            check(!cosmo_webgpu_configure("embedded", "auto"), "configure failed");
            WGPUChainedStruct tail{}; expected_tail = &tail;
            WGPUInstanceDescriptor descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT; descriptor.nextInChain = &tail;
            check(cosmo_webgpu_create_instance("embedded", &descriptor) != nullptr, "mocked factory failed");
            check(registration_calls == 1 && selection_calls == 1 && factory_calls == 1, "factory sequence mismatch");
            check(factory_provider == "embedded", "wrong provider selected");
            check(!cosmo_webgpu_configure("embedded", "auto"), "identical initialized policy must be idempotent");
            check(cosmo_webgpu_configure("native", "auto") != 0, "initialized provider changed");
            check(cosmo_webgpu_configure("embedded", "WebGPU1") != 0, "initialized device changed");
            check(!cosmo_webgpu_create_instance("native", nullptr), "different factory provider accepted");
            check(factory_calls == 1, "rejected factory still called WGPU");
        } else if (scenario == "native-thread-guard") {
            check(!cosmo_webgpu_configure("native", "auto"), "configure failed");
            pthread_t thread; void *result = reinterpret_cast<void *>(1);
            check(pthread_create(&thread, nullptr, native_from_thread, &result) == 0, "pthread_create failed");
            check(pthread_join(thread, nullptr) == 0, "pthread_join failed");
            if (IsLinux()) {
                check(!result && factory_calls == 0 && registration_calls == 0, "Linux nonmain factory escaped guard");
                check(strstr(cosmo_webgpu_last_error(), "original main thread"), "missing thread diagnostic");
            } else {
                check(result != nullptr && factory_calls == 1, "non-Linux mocked factory unexpectedly restricted");
            }
        } else throw std::runtime_error("unknown scenario");
        std::printf("{\"status\":\"PASS\",\"scenario\":\"%s\",\"scope\":\"real policy with mocked Vulkan/WGPU factory\",\"linux\":%s,\"factory_calls\":%d}\n",
                    argv[1], IsLinux() ? "true" : "false", factory_calls);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "DEVICE_POLICY_TEST FAIL: %s\n", error.what()); return 1;
    }
}
