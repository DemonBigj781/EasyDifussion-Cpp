/* SPDX-License-Identifier: MIT */
#include "inference_worker.hpp"
#include "logging.h"

#include <cstdio>
#include <exception>
#include <pthread.h>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace {
constexpr size_t inference_stack_bytes = 8 * 1024 * 1024;

struct Work {
    std::function<std::vector<std::string>()> function;
    std::vector<std::string> result;
    std::exception_ptr failure;
    bool out_of_vram = false;
};

void require_pthread(int error, const char * operation) {
    if (error) throw std::system_error(error, std::generic_category(), operation);
}

void * run(void * opaque) noexcept {
    auto & work = *static_cast<Work *>(opaque);
    reset_sd_generation_error();
    try {
        pthread_attr_t actual;
        require_pthread(pthread_getattr_np(pthread_self(), &actual), "inference pthread_getattr_np");
        size_t bytes = 0, guard = 0;
        const int stack_error = pthread_attr_getstacksize(&actual, &bytes);
        const int guard_error = pthread_attr_getguardsize(&actual, &guard);
        const int destroy_error = pthread_attr_destroy(&actual);
        require_pthread(stack_error, "inference pthread_attr_getstacksize");
        require_pthread(guard_error, "inference pthread_attr_getguardsize");
        require_pthread(destroy_error, "inference pthread_attr_destroy(actual)");
        if (bytes < inference_stack_bytes)
            throw std::runtime_error("Native inference worker stack is smaller than requested");
        std::fprintf(stderr, "NATIVE_INFERENCE_WORKER stack_bytes=%zu guard_bytes=%zu "
                             "stack_source=pthread_getattr_np\n", bytes, guard);
        std::fflush(stderr);
        work.result = work.function();
    } catch (...) {
        /* Preserve the original exception type for the HTTP handler's existing
           invalid_argument/standard-exception response classification. */
        work.failure = std::current_exception();
    }
    work.out_of_vram = sd_generation_error_out_of_vram();
    return nullptr;
}
}

std::vector<std::string> cosmo_inference_worker(
    std::function<std::vector<std::string>()> function) {
    Work work{std::move(function), {}, {}, false};
    pthread_attr_t attributes;
    require_pthread(pthread_attr_init(&attributes), "inference pthread_attr_init");
    const int stack_error = pthread_attr_setstacksize(&attributes, inference_stack_bytes);
    if (stack_error) {
        pthread_attr_destroy(&attributes);
        require_pthread(stack_error, "inference pthread_attr_setstacksize");
    }
    pthread_t worker;
    const int create_error = pthread_create(&worker, &attributes, run, &work);
    const int destroy_error = pthread_attr_destroy(&attributes);
    require_pthread(create_error, "inference pthread_create");
    /* Always join once a thread exists, including if attribute destruction
       failed. Returning early would invalidate the worker's captured request. */
    const int join_error = pthread_join(worker, nullptr);
    if (join_error) {
        std::fprintf(stderr, "Fatal native inference pthread_join error: %d\n", join_error);
        std::terminate();
    }
    set_sd_generation_error_out_of_vram(work.out_of_vram);
    require_pthread(destroy_error, "inference pthread_attr_destroy");
    if (work.failure) std::rethrow_exception(work.failure);
    return std::move(work.result);
}
