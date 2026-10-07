/* SPDX-License-Identifier: MIT */
#include "runtime.h"
#include "cosmo-webgpu.h"
#include "ggml-backend.h"
#include "native_main_executor.hpp"
#include <cstdio>
#include <cstring>
#include <exception>

extern "C" int cosmo_webgpu_device_graph_probe(const char *selector, int require_hardware);

extern "C" int cosmo_webgpu_device_test(int argc, char **argv) {
    int require_hardware = 0;
    bool all = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--require-hardware")) require_hardware = 1;
        else if (!std::strcmp(argv[i], "--all")) all = true;
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            std::puts("Usage: webgpu-device-test [--provider auto|native|embedded] [--device NAME] [--require-hardware | --all]\n"
                      "Runs four actual F32/Q4_0 graphs and twelve independent scalar readback checks.\n"
                      "Hardware requires a reported integrated/discrete GPU; software and unknown adapters fail that gate.");
            return 0;
        } else {
            std::fprintf(stderr, "webgpu-device-test: unknown argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (all && require_hardware) {
        std::fputs("webgpu-device-test: --all and --require-hardware are mutually exclusive\n", stderr);
        return 2;
    }
    try {
        if (cosmo_webgpu_initialize()) return 1;
        // Discover/create devices before moving any work to a service thread.
        (void)ggml_backend_reg_by_name("WebGPU");
        int result = 1;
        auto probe = [&] {
            if (!all) {
                result = cosmo_webgpu_device_graph_probe(cosmo_webgpu_requested_device(), require_hardware);
                return;
            }
            const size_t count = cosmo_webgpu_device_count();
            if (!count) {
                std::fputs("WEBGPU_DEVICE_ALL FAIL: no compatible adapters\n", stderr);
                return;
            }
            result = 0;
            for (size_t i = 0; i < count; ++i) {
                const auto *info = cosmo_webgpu_device_at(i);
                if (!info) { result = 1; break; }
                std::printf("WEBGPU_DEVICE_BEGIN selector=%s\n", info->selector);
                const int current = cosmo_webgpu_device_graph_probe(info->selector, 0);
                std::printf("WEBGPU_DEVICE_END selector=%s result=%d\n", info->selector, current);
                if (current) { result = current; break; }
            }
            std::printf("WEBGPU_DEVICE_ALL count=%zu %s\n", count, result ? "FAIL" : "PASS");
        };
        if (cosmo_native_main_required()) {
            cosmo_native_main_service([&] {
                cosmo_native_main_invoke(probe);
            });
            std::puts("WEBGPU_DEVICE_LANE mode=main-thread-service");
        } else {
            probe();
            std::puts("WEBGPU_DEVICE_LANE mode=direct");
        }
        return result;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "WEBGPU_DEVICE_TEST FAIL: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("WEBGPU_DEVICE_TEST FAIL: unknown native exception\n", stderr);
        return 1;
    }
}
