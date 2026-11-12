#ifndef __SD_RUNTIME_LORA_APPLY_POLICY_H__
#define __SD_RUNTIME_LORA_APPLY_POLICY_H__

#include "ggml.h"

namespace sd {

template <typename StorageMap>
inline bool lora_auto_merge_allowed(const StorageMap& tensors,
                                    bool have_quantized_weight,
                                    bool streaming_constrained,
                                    bool row_split_active) {
    if (have_quantized_weight || streaming_constrained || row_split_active) {
        return false;
    }
    for (const auto& entry : tensors) {
        const auto type = entry.second.expected_type;
        if (static_cast<int>(type) >= 0 && type < GGML_TYPE_COUNT && ggml_is_quantized(type)) {
            return false;
        }
    }
    return true;
}

}  // namespace sd

#endif  // __SD_RUNTIME_LORA_APPLY_POLICY_H__
