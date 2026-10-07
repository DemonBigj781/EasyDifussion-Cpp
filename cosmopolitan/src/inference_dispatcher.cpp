/* SPDX-License-Identifier: MIT */
#include "inference_dispatcher.hpp"

#include <cstdio>
#include <exception>
#include <utility>

CosmoInferenceDispatcher::CosmoInferenceDispatcher()
    : coordinator_([this] { run(); }) {}

CosmoInferenceDispatcher::~CosmoInferenceDispatcher() { shutdown(); }

bool CosmoInferenceDispatcher::try_submit(std::function<void()> work, std::function<void()> deliver) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (busy_ || closed_) return false;
    work_ = std::move(work);
    deliver_ = std::move(deliver);
    busy_ = true;
    changed_.notify_one();
    return true;
}

bool CosmoInferenceDispatcher::busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return busy_ || closed_;
}

void CosmoInferenceDispatcher::close_admission() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
}

void CosmoInferenceDispatcher::shutdown() {
    std::lock_guard<std::mutex> joining(shutdown_mutex_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = stopping_ = true;
        changed_.notify_one();
    }
    if (coordinator_.joinable()) coordinator_.join();
}

void CosmoInferenceDispatcher::run() noexcept {
    for (;;) {
        std::function<void()> work, deliver;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [this] { return stopping_ || static_cast<bool>(work_); });
            if (stopping_) {
                work_ = deliver_ = {};
                busy_ = false;
                return;
            }
            work = std::move(work_);
            deliver = std::move(deliver_);
            work_ = {};
            deliver_ = {};
        }
        try {
            work();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                busy_ = false;
            }
            deliver();
        } catch (...) {
            /* A job must translate its own native failures to a response.
               Silently losing a completion would leave an HTTP request hung. */
            std::fputs("Fatal exception in native HTTP inference dispatcher\n", stderr);
            std::terminate();
        }
    }
}
