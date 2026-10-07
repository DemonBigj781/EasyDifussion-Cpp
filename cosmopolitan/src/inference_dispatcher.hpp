/* SPDX-License-Identifier: MIT */
#ifndef COSMO_INFERENCE_DISPATCHER_HPP
#define COSMO_INFERENCE_DISPATCHER_HPP

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

/* One bounded coordinator keeps long synchronous generation off Crow's I/O
   loops. Work must capture native errors; deliver only posts completion to the
   appropriate I/O loop. All captures remain owned until work/delivery return. */
class CosmoInferenceDispatcher {
public:
    CosmoInferenceDispatcher();
    ~CosmoInferenceDispatcher();
    CosmoInferenceDispatcher(const CosmoInferenceDispatcher &) = delete;
    CosmoInferenceDispatcher & operator=(const CosmoInferenceDispatcher &) = delete;

    bool try_submit(std::function<void()> work, std::function<void()> deliver);
    bool busy() const;
    void close_admission();
    /* Call outside I/O handlers, before server dependencies are destroyed. */
    void shutdown();

private:
    void run() noexcept;
    mutable std::mutex mutex_;
    std::mutex shutdown_mutex_;
    std::condition_variable changed_;
    std::function<void()> work_, deliver_;
    bool busy_ = false, closed_ = false, stopping_ = false;
    std::thread coordinator_;
};

#endif
