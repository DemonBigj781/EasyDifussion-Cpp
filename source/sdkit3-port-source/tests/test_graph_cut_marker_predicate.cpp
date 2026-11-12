#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#include "core/ggml_graph_cut.h"

static size_t allocations = 0;
void* operator new(size_t size) {
    ++allocations;
    if (void* ptr = std::malloc(size ? size : 1)) return ptr;
    throw std::bad_alloc();
}
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, size_t) noexcept { std::free(ptr); }

int main() {
    using namespace sd::ggml_graph_cut;
    std::vector<ggml_tensor> tensors;
    std::vector<bool> expected;
    const std::string prefix = GGML_RUNNER_CUT_PREFIX, suffix = GGML_RUNNER_CUT_SUFFIX;
    auto add = [&](const std::string& name) {
        ggml_tensor tensor{};
        ggml_set_name(&tensor, name.c_str());
        const std::string stored = tensor.name;
        tensors.push_back(tensor);
        expected.push_back(stored.size() >= prefix.size() && stored.size() >= suffix.size() &&
                           stored.compare(0, prefix.size(), prefix) == 0 &&
                           stored.compare(stored.size() - suffix.size(), suffix.size(), suffix) == 0);
    };
    for (size_t length = 0; length <= GGML_MAX_NAME + 32; ++length) {
        const std::string body(length, 'x');
        add(body); add(prefix + body); add(prefix + body + suffix);
        add(body + suffix); add("!" + prefix + body + suffix); add(prefix + body + suffix + "!");
    }
    add(""); add(prefix); add(suffix); add(prefix + suffix); add(prefix + "group|output|");
    add(prefix + "\ngroup|output|");
    if (is_graph_cut_tensor(nullptr)) return 1;
    const auto before = allocations;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (is_graph_cut_tensor(&tensors[i]) != expected[i]) {
            std::fprintf(stderr, "marker mismatch at case %zu\n", i);
            return 1;
        }
    }
    const auto observed = allocations - before;
    std::printf("%zu marker cases match legacy semantics; %zu predicate heap allocations\n", tensors.size(), observed);
    if (observed != 0) return 1;
    return 0;
}
