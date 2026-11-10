#include "overflow.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <utility>

namespace edcpp::api::overflow {
namespace {

constexpr std::size_t backend_slots =
    static_cast<std::size_t>(Backend::directml) + 1;

std::array<const Translation*, backend_slots>& translations() {
    static std::array<const Translation*, backend_slots> table{};
    return table;
}

} // namespace

bool register_translation(const Translation* translation) noexcept {
    if (translation == nullptr || translation->backend == Backend::none ||
        translation->prepare == nullptr ||
        translation->allocate == nullptr ||
        translation->release == nullptr) {
        return false;
    }
    const auto index = static_cast<std::size_t>(translation->backend);
    if (index >= backend_slots) {
        return false;
    }
    translations()[index] = translation;
    return true;
}

const Translation* translation_for(Backend backend) noexcept {
    const auto index = static_cast<std::size_t>(backend);
    return backend != Backend::none && index < backend_slots
        ? translations()[index]
        : nullptr;
}

Result allocate(Backend backend, const Request& request) {
    const Translation* translation = translation_for(backend);
    if (translation == nullptr) {
        Result result;
        result.diagnostic =
            "Common Overflow has no registered translation for the requested backend";
        return result;
    }

    PlanningResult prepared = translation->prepare(request);
    if (!prepared.ready) {
        Result result;
        result.diagnostic = prepared.diagnostic.empty()
            ? "Common Overflow could not prepare a normalized allocation plan"
            : std::move(prepared.diagnostic);
        return result;
    }

    bool primary_out_of_memory = false;
    std::string failures;
    const auto append_failure = [&failures](const std::string& message) {
        if (message.empty()) return;
        if (!failures.empty()) failures += "; ";
        failures += message;
    };

    for (const Candidate& candidate : plan(prepared.input, request.size)) {
        const Translation* owner = translation_for(candidate.backend);
        if (owner == nullptr) {
            append_failure(
                "Common Overflow has no translation for a planned candidate");
            continue;
        }

        Result attempt = normalize(owner->allocate(request, candidate));
        if (candidate.tier == Tier::active_vram) {
            primary_out_of_memory = attempt.primary_out_of_memory;
        }
        if (attempt.allocated) {
            attempt.primary_out_of_memory = primary_out_of_memory;
            attempt.overflowed = candidate.tier != Tier::active_vram;
            return attempt;
        }

        append_failure(attempt.diagnostic);
        if (candidate.tier == Tier::active_vram &&
            !attempt.primary_out_of_memory) {
            Result result;
            result.diagnostic = std::move(failures);
            return result;
        }
    }

    Result result;
    result.primary_out_of_memory = primary_out_of_memory;
    result.diagnostic = failures.empty()
        ? "Common Overflow found no implemented allocation candidate"
        : std::move(failures);
    return result;
}

ReleaseResult release(Resource& resource) {
    const Translation* translation = translation_for(resource.backend);
    if (translation == nullptr) {
        return {
            false,
            "Common Overflow has no registered translation for the resource owner"};
    }
    ReleaseResult result =
        normalize(translation->release(resource), resource);
    if (result.released) {
        resource = {};
    }
    return result;
}

} // namespace edcpp::api::overflow
