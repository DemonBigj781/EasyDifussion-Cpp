/* SPDX-License-Identifier: MIT */
#include "native_main_executor.hpp"
#include "cosmo-webgpu.h"
#include <cosmo.h>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace {
constexpr size_t service_stack_bytes = 8 * 1024 * 1024;
struct Job {
    std::function<void()> work;
    std::exception_ptr failure;
    bool done = false;
};
struct Lane {
    std::mutex mutex;
    std::condition_variable changed;
    std::shared_ptr<Job> pending;
    pthread_t owner;
    std::function<void()> service;
    std::exception_ptr service_failure;
    bool service_done = false;
};
std::mutex lane_mutex;
std::shared_ptr<Lane> active_lane;

void require_pthread(int error, const char *operation) {
    if (error) throw std::system_error(error, std::generic_category(), operation);
}
void *run_service(void *opaque) noexcept {
    Lane &lane = *static_cast<Lane *>(opaque);
    try {
        pthread_attr_t actual;
        require_pthread(pthread_getattr_np(pthread_self(), &actual), "native HTTP pthread_getattr_np");
        size_t bytes = 0, guard = 0;
        const int stack_error = pthread_attr_getstacksize(&actual, &bytes);
        const int guard_error = pthread_attr_getguardsize(&actual, &guard);
        const int destroy_error = pthread_attr_destroy(&actual);
        require_pthread(stack_error, "native HTTP pthread_attr_getstacksize");
        require_pthread(guard_error, "native HTTP pthread_attr_getguardsize");
        require_pthread(destroy_error, "native HTTP pthread_attr_destroy(actual)");
        if (bytes < service_stack_bytes) throw std::runtime_error("Native HTTP service stack is too small");
        std::fprintf(stderr, "NATIVE_HTTP_SERVICE stack_bytes=%zu guard_bytes=%zu "
                             "stack_source=pthread_getattr_np\n", bytes, guard);
        std::fflush(stderr);
        lane.service();
    }
    catch (...) { lane.service_failure = std::current_exception(); }
    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        lane.service_done = true;
    }
    lane.changed.notify_all();
    return nullptr;
}
}

bool cosmo_native_main_required() {
    return IsLinux() && std::strcmp(cosmo_webgpu_requested_provider(), "embedded") != 0;
}

bool cosmo_native_model_load_on_main(const std::function<void()> &work) {
    if (!cosmo_native_main_required()) return false;
    if (gettid() != getpid())
        throw std::runtime_error("Native Vulkan model loading requires the original main thread");
    work();
    return true;
}

void cosmo_native_main_invoke(std::function<void()> work) {
    std::shared_ptr<Lane> lane;
    {
        std::lock_guard<std::mutex> lock(lane_mutex);
        lane = active_lane;
    }
    if (!lane) throw std::runtime_error("Native Vulkan requires the original main-thread executor");
    if (pthread_equal(pthread_self(), lane->owner)) {
        work();
        return;
    }
    auto job = std::make_shared<Job>();
    job->work = std::move(work);
    std::unique_lock<std::mutex> lock(lane->mutex);
    if (lane->service_done || lane->pending)
        throw std::runtime_error("Native main-thread executor is closed or busy");
    lane->pending = job;
    lane->changed.notify_all();
    lane->changed.wait(lock, [&] { return job->done; });
    if (job->failure) std::rethrow_exception(job->failure);
}

void cosmo_native_main_service(std::function<void()> service) {
    if (!cosmo_native_main_required()) { service(); return; }
    if (gettid() != getpid())
        throw std::runtime_error("Native Vulkan service must start on the original main thread");
    auto lane = std::make_shared<Lane>();
    lane->owner = pthread_self();
    lane->service = std::move(service);
    {
        std::lock_guard<std::mutex> lock(lane_mutex);
        if (active_lane) throw std::runtime_error("Native main-thread executor is already running");
        active_lane = lane;
    }
    pthread_attr_t attributes;
    int error = pthread_attr_init(&attributes);
    if (error) {
        std::lock_guard<std::mutex> lock(lane_mutex); active_lane.reset();
        require_pthread(error, "native HTTP pthread_attr_init");
    }
    const int stack_error = pthread_attr_setstacksize(&attributes, service_stack_bytes);
    pthread_t thread;
    const int create_error = stack_error ? stack_error : pthread_create(&thread, &attributes, run_service, lane.get());
    const int destroy_error = pthread_attr_destroy(&attributes);
    if (create_error) {
        std::lock_guard<std::mutex> lock(lane_mutex); active_lane.reset();
        require_pthread(create_error, "native HTTP pthread_create/stack");
    }
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(lane->mutex);
            lane->changed.wait(lock, [&] { return lane->pending || lane->service_done; });
            if (!lane->pending) break;
            job = lane->pending;
        }
        try { job->work(); }
        catch (...) { job->failure = std::current_exception(); }
        {
            std::lock_guard<std::mutex> lock(lane->mutex);
            job->done = true;
            lane->pending.reset();
        }
        lane->changed.notify_all();
    }
    const int join_error = pthread_join(thread, nullptr);
    if (join_error) {
        std::fprintf(stderr, "Fatal native HTTP pthread_join error: %d\n", join_error);
        std::terminate();
    }
    {
        std::lock_guard<std::mutex> lock(lane_mutex); active_lane.reset();
    }
    require_pthread(destroy_error, "native HTTP pthread_attr_destroy");
    if (lane->service_failure) std::rethrow_exception(lane->service_failure);
}
