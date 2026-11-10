#include "api/cpu_byte_buffer.hpp"
#include "native/cpu/byte_storage.hpp"
#include "features/unload/common/devices/unload.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace edcpp::api;
static_assert(!std::is_copy_constructible_v<easyapi::CpuByteBuffer>);
static_assert(!std::is_copy_assignable_v<easyapi::CpuByteBuffer>);
static_assert(std::is_nothrow_move_constructible_v<easyapi::CpuByteBuffer>);
static_assert(std::is_nothrow_move_assignable_v<easyapi::CpuByteBuffer>);
static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static void noop_rollback(load::Resource&) noexcept {}
static void reject_load(const load::Request&, load::Result&) {}
static unload::Result reject_unload(const load::Resource&) { return {}; }

int main() {
    try {
        const std::array<unsigned char, 6> bytes{0, 1, 2, 127, 254, 255};
        easyapi::CpuByteBuffer empty;
        empty.reset();
        require(empty.empty() && empty.data() == nullptr && empty.size() == 0, "default owner not empty");
        auto first = easyapi::CpuByteBuffer::copy_from(bytes.data(), bytes.size());
        const auto* original = first.data();
        auto second = std::move(first);
        require(first.empty() && second.data() == original, "move construction duplicated ownership");
        auto third = easyapi::CpuByteBuffer::copy_from(bytes.data(), bytes.size());
        third = std::move(second);
        require(second.empty() && third.data() == original, "move assignment duplicated ownership");
        auto* same_owner = &third;
        third = std::move(*same_owner);
        require(third.data() == original, "self move discarded ownership");
        third.reset();
        third.reset();
        require(third.empty(), "reset retained ownership");
        bool rejected = false;
        try { (void)easyapi::CpuByteBuffer::copy_from(nullptr, 0); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "empty payload was accepted");

        auto module = std::make_shared<int>(42);
        std::weak_ptr<int> module_lifetime = module;
        require(load::register_model_translation(Backend::mesa, &reject_load, &noop_rollback, module), "module load registration failed");
        require(unload::register_model_translation(Backend::mesa, &reject_unload, module), "module unload registration failed");
        module.reset();
        require(!module_lifetime.expired(), "registry released module callbacks too soon");
        load::Request dummy{bytes.data(), bytes.size(), 0, load::ResourceType::model};
        require(!load::load_resource(Backend::mesa, dummy).loaded, "fixture backend unexpectedly allocated");
        require(!module_lifetime.expired(), "dispatch released module lifetime");

        std::atomic<int> failures{0}, cycles{0};
        std::vector<std::thread> threads;
        for (int worker = 0; worker < 8; ++worker) {
            threads.emplace_back([&] {
                try {
                    for (int i = 0; i < 128; ++i) {
                        // Re-registration must never replace the release operation
                        // of a concurrently alive resource.
                        if (load::register_model_translation(Backend::cpu, &reject_load, &noop_rollback)) ++failures;
                        if (unload::register_model_translation(Backend::cpu, &reject_unload)) ++failures;
                        auto buffer = easyapi::CpuByteBuffer::copy_from(bytes.data(), bytes.size());
                        auto moved = std::move(buffer);
                        if (!buffer.empty() || moved.size() != bytes.size() ||
                            std::memcmp(moved.data(), bytes.data(), bytes.size()) != 0) ++failures;
                        ++cycles;
                    }
                } catch (...) { ++failures; }
            });
        }
        for (auto& thread : threads) thread.join();
        require(failures == 0 && cycles == 1024, "concurrent ownership/registry failure");
        require(edcpp::native::cpu::live_allocations() == 0, "owned allocations leaked");
        std::cout << "CPU exclusive ownership: moves, reset and 1024 concurrent cycles passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
