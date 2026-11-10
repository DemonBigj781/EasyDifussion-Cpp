#ifndef SDKIT_GENERATION_ADMISSION_H
#define SDKIT_GENERATION_ADMISSION_H

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

// Admission covers model selection as well as execution. Interrupt uses a
// separate identity lock, held through forwarding, and never takes the gate.
class GenerationAdmission {
public:
    enum class InterruptResult { forwarded, no_active, mismatch, unavailable };
    class Lease {
    public:
        explicit Lease(GenerationAdmission& owner)
            : owner_(owner), lock_(owner.gate_, std::try_to_lock) {}
        ~Lease() {
            if (bound_) {
                std::lock_guard<std::mutex> identity(owner_.identity_);
                owner_.task_.clear();
            }
        }
        bool acquired() const { return lock_.owns_lock(); }
        void bind(const std::string& task) {
            if (!acquired() || task.empty()) throw std::logic_error("Invalid generation admission");
            std::lock_guard<std::mutex> identity(owner_.identity_);
            owner_.task_ = task;
            bound_ = true;
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
    private:
        GenerationAdmission& owner_;
        std::unique_lock<std::mutex> lock_;
        bool bound_ = false;
    };
    Lease tryAcquire() { return Lease(*this); }

    template<class Forward>
    InterruptResult interrupt(const std::optional<std::string>& requested, Forward&& forward) {
        std::lock_guard<std::mutex> identity(identity_);
        if (task_.empty()) return InterruptResult::no_active;
        if (requested && *requested != task_) return InterruptResult::mismatch;
        return forward(task_) ? InterruptResult::forwarded : InterruptResult::unavailable;
    }
private:
    std::mutex gate_;
    std::mutex identity_;
    std::string task_;
};
#endif
