/* SPDX-License-Identifier: MIT */
#include "llama_service.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace {
void cleanup(cosmo_llama_result *result) noexcept {
    // The engine records each owned resource before its next throwing call.
    // Its cleanup clears model/context pointers before invoking destructors.
    // Ordinary llama destructors do not throw; keep the C API guarded even
    // if an unexpected upstream exception occurs during recovery.
    for (int attempt = 0; result && result->_state && attempt < 3; ++attempt) {
        try { cosmo_llama_service_cleanup(result); }
        catch (...) {}
    }
}
}

extern "C" int cosmo_llama_service_generate(const cosmo_llama_request *request,
                                            cosmo_llama_result *result) noexcept {
    if (!result) return COSMO_LLAMA_INVALID;
    if (result->text || result->token_ids || result->_state) {
        std::snprintf(result->error, sizeof(result->error), "Free the previous text result before reuse");
        return COSMO_LLAMA_INVALID;
    }
    *result = {};
    if (!request) {
        std::snprintf(result->error, sizeof(result->error), "Text inference request is required");
        return COSMO_LLAMA_INVALID;
    }
    try {
        const int status = cosmo_llama_service_engine(request, result);
        if (status != COSMO_LLAMA_OK && !result->error[0])
            std::snprintf(result->error, sizeof(result->error), "Text inference failed");
        return status;
    } catch (const std::exception &error) {
        std::snprintf(result->error, sizeof(result->error), "Text inference exception: %s", error.what());
    } catch (...) {
        std::snprintf(result->error, sizeof(result->error), "Unknown text inference exception");
    }
    cleanup(result);
    return COSMO_LLAMA_ERROR;
}

extern "C" void cosmo_llama_result_free(cosmo_llama_result *result) noexcept {
    if (!result) return;
    cleanup(result);
    std::free(result->text);
    std::free(result->token_ids);
    *result = {};
}
