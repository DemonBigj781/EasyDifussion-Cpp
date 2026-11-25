#include "generation_admission.h"
#include "task_state.h"
#include <atomic>
#include <condition_variable>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    try {
        TaskStateManager tasks;
        tasks.createTask("partial");
        tasks.updateTaskProgress("partial", .25f, "", 2, 8);
        tasks.interruptTask("partial");
        tasks.completeTask("partial", {}, "{}");
        const auto partial = tasks.getTaskProgressState("partial", false, -1);
        require(partial.completed && partial.interrupted && partial.current_step == 2 && partial.progress == .25f,
                "interrupted completion fabricated full sampling progress");
        GenerationAdmission admission;
        using Result = GenerationAdmission::InterruptResult;
        int forwarded = 0;
        auto callback = [&](const std::string& task) { require(task == "default_task", "wrong identity"); ++forwarded; return true; };
        require(admission.interrupt(std::nullopt, callback) == Result::no_active, "idle cancellation forwarded");
        for (int repetition = 0; repetition < 2; ++repetition) {
            auto generation = admission.tryAcquire();
            require(generation.acquired(), "sequential default task rejected");
            generation.bind("default_task");
            // All mutating routes use this same admission gate before touching resources.
            std::thread contender([&] {
                for (int operation = 0; operation < 5; ++operation) {
                    auto other = admission.tryAcquire();
                    if (other.acquired()) std::terminate();
                }
            });
            contender.join();
            require(admission.interrupt(std::string("old-task"), callback) == Result::mismatch, "stale cancellation forwarded");
            require(admission.interrupt(std::string("default_task"), [](const std::string&) { return false; }) == Result::unavailable,
                    "preparation falsely acknowledged cancellation");
            require(admission.interrupt(std::string("default_task"), callback) == Result::forwarded, "matching task not forwarded");
            require(admission.interrupt(std::nullopt, callback) == Result::forwarded, "legacy cancellation changed");
        }
        require(forwarded == 4, "unexpected callback count");
        try {
            auto failed = admission.tryAcquire();
            failed.bind("failed-task");
            throw std::runtime_error("simulated load failure");
        } catch (const std::runtime_error&) {}
        require(admission.interrupt(std::string("failed-task"), callback) == Result::no_active, "exception retained identity");
        { auto next = admission.tryAcquire(); require(next.acquired(), "exception retained gate"); }

        // Completion must not clear ownership and admit a successor while a
        // matching interrupt callback is still selecting the current context.
        std::promise<void> bound, callback_started, finish_generation, release_callback;
        auto finish = finish_generation.get_future();
        auto release = release_callback.get_future();
        std::atomic<bool> completion_started{false}, completion_done{false};
        std::thread owner([&] {
            { auto lease = admission.tryAcquire(); lease.bind("owned"); bound.set_value(); finish.wait(); completion_started = true; }
            completion_done = true;
        });
        bound.get_future().wait();
        std::thread interrupt([&] {
            auto result = admission.interrupt(std::string("owned"), [&](const std::string&) {
                callback_started.set_value(); release.wait(); return true;
            });
            if (result != Result::forwarded) std::terminate();
        });
        callback_started.get_future().wait();
        finish_generation.set_value();
        while (!completion_started) std::this_thread::yield();
        { auto next = admission.tryAcquire(); require(!next.acquired(), "successor entered during cancellation"); }
        require(!completion_done, "identity cleared during callback");
        release_callback.set_value(); interrupt.join(); owner.join();
        { auto next = admission.tryAcquire(); require(next.acquired(), "completion did not release gate"); next.bind("successor");
          require(admission.interrupt(std::string("owned"), callback) == Result::mismatch, "old callback reached successor"); }
        std::cout << "Admission, exception unwind, legacy and scoped cancellation passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
