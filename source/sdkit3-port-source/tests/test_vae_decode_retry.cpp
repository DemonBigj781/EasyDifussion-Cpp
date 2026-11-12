#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "ggml-cpu.h"
#include "runtime/vae_decode_retry.h"

static bool fail_reserve = false;
extern "C" bool __real_ggml_gallocr_reserve(ggml_gallocr_t, ggml_cgraph*);
extern "C" bool __wrap_ggml_gallocr_reserve(ggml_gallocr_t allocator, ggml_cgraph* graph) {
    return fail_reserve ? false : __real_ggml_gallocr_reserve(allocator, graph);
}
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

class RetryVAE : public FakeVAE {
public:
    enum Mode { FINITE, NAN_OUTPUT, INF_OUTPUT, RESERVE_FAILURE, BAD_READBACK, EMPTY_WITHOUT_COMPUTE };
    Mode mode = FINITE;
    int calls = 0;
    int fail_on_call = 1;
    int fail_through_call = 1000000;
    bool cancel = false;
    bool cancel_after_compute = false;
    explicit RetryVAE(ggml_backend_t backend) : FakeVAE(VERSION_CHROMA_RADIANCE, backend) {
        set_execution_cancel_callback([this] { return cancel; });
    }
    bool supports_temporal_tiling(VAETemporalDirection) const override { return true; }
    sd::Tensor<float> _compute(int threads, const sd::Tensor<float>& z, bool) override {
        ++calls;
        const bool inject = calls >= fail_on_call && calls <= fail_through_call;
        if (inject && mode == EMPTY_WITHOUT_COMPUTE) return {};
        fail_reserve = inject && mode == RESERVE_FAILURE;
        sd::Tensor<int32_t> integers(z.shape());
        auto result = compute<float>([&] {
            auto graph = new_graph_custom(64);
            auto out = inject && mode == BAD_READBACK ? ggml_dup(compute_ctx, make_input(integers))
                                                     : ggml_scale(compute_ctx, make_input(z), 1.f);
            ggml_build_forward_expand(graph, out);
            return graph;
        }, threads);
        fail_reserve = false;
        if (cancel_after_compute) cancel = true;
        if (!result) return {};
        if (inject && mode == NAN_OUTPUT) result->values()[0] = std::numeric_limits<float>::quiet_NaN();
        if (inject && mode == INF_OUTPUT) result->values()[0] = std::numeric_limits<float>::infinity();
        return std::move(*result);
    }
};

class RecordingRetryVAE : public RetryVAE {
public:
    std::vector<sd_tiling_params_t> attempts;
    bool flags_forwarded = true;
    bool expect_video = false;
    explicit RecordingRetryVAE(ggml_backend_t backend) : RetryVAE(backend) {}
    sd::Tensor<float> decode(int threads, const sd::Tensor<float>& input, sd_tiling_params_t tiling,
                            bool video, bool circular_x, bool circular_y, bool silent = false) override {
        attempts.push_back(tiling);
        flags_forwarded = flags_forwarded && video == expect_video && circular_x && !circular_y;
        return VAE::decode(threads, input, tiling, video, circular_x, circular_y, true);
    }
};

static void test_retry_loop(ggml_backend_t backend) {
    const sd::Tensor<float> input({8, 8, 3, 1});
    auto run = [&](RecordingRetryVAE& vae, sd_tiling_params_t& tiling, bool auto_fit, bool video,
                   bool cancel_between_attempts = false) {
        vae.expect_video = video;
        return sd::decode_vae_with_retries(vae, 1, input, tiling, auto_fit, video, true, false,
                                          [&] { return cancel_between_attempts && !vae.attempts.empty(); });
    };
    for (auto failure : {RetryVAE::RESERVE_FAILURE, RetryVAE::NAN_OUTPUT, RetryVAE::INF_OUTPUT}) {
        RecordingRetryVAE vae(backend);
        vae.mode = failure;
        vae.fail_through_call = 1;
        sd_tiling_params_t tiling{};
        require(!run(vae, tiling, false, false).empty(), "legacy image fallback did not recover");
        require(vae.attempts.size() == 2 && !vae.attempts[0].enabled && vae.attempts[1].enabled &&
                    vae.attempts[1].tile_size_x == 32 && vae.attempts[1].tile_size_y == 32 &&
                    !tiling.enabled && vae.flags_forwarded,
                "legacy fallback shape, flags, retry bound or caller parameters changed");
    }
    {
        RecordingRetryVAE vae(backend);
        vae.mode = RetryVAE::RESERVE_FAILURE;
        sd_tiling_params_t tiling{};
        require(run(vae, tiling, true, true).empty() && vae.attempts.size() == 3,
                "persistent video allocation failure was not bounded at three attempts");
        require(!vae.attempts[0].enabled && !vae.attempts[0].temporal_tiling &&
                    !vae.attempts[1].enabled && vae.attempts[1].temporal_tiling &&
                    vae.attempts[2].enabled && vae.attempts[2].temporal_tiling && vae.flags_forwarded,
                "auto-fit temporal-first ordering changed");
    }
    for (bool auto_fit : {false, true}) {
        RecordingRetryVAE vae(backend);
        vae.mode = RetryVAE::RESERVE_FAILURE;
        sd_tiling_params_t tiling{};
        require(run(vae, tiling, auto_fit, false, true).empty() && vae.attempts.size() == 1 && !tiling.enabled,
                "cancellation between attempts triggered another decode or changed tiling");
    }
    for (auto failure : {RetryVAE::BAD_READBACK, RetryVAE::EMPTY_WITHOUT_COMPUTE}) {
        RecordingRetryVAE vae(backend);
        vae.mode = failure;
        sd_tiling_params_t tiling{};
        require(run(vae, tiling, true, false).empty() && vae.attempts.size() == 1 && !tiling.enabled,
                "non-allocation error entered retry loop");
    }
    {
        RecordingRetryVAE vae(backend);
        vae.mode = RetryVAE::RESERVE_FAILURE;
        sd_tiling_params_t tiling{};
        require(run(vae, tiling, true, false).empty() && vae.attempts.size() == 2 && tiling.enabled,
                "auto-fit image failure exceeded the single spatial retry");
    }
    {
        RecordingRetryVAE vae(backend);
        vae.mode = RetryVAE::RESERVE_FAILURE;
        sd_tiling_params_t tiling{};
        tiling.enabled = true;
        tiling.tile_size_x = tiling.tile_size_y = 4;
        require(run(vae, tiling, false, false).empty() && vae.attempts.size() == 1,
                "already-tiled image retried without an available policy transition");
    }
    {
        RecordingRetryVAE vae(backend);
        vae.mode = RetryVAE::RESERVE_FAILURE;
        vae.fail_through_call = 1;
        sd_tiling_params_t tiling{};
        tiling.tile_size_x = 6;
        tiling.tile_size_y = 7;
        tiling.target_overlap = 0.25f;
        require(!run(vae, tiling, false, false).empty() && vae.attempts.size() == 2 &&
                    vae.attempts[1].tile_size_x == 6 && vae.attempts[1].tile_size_y == 7 &&
                    vae.attempts[1].target_overlap == 0.25f && !tiling.enabled,
                "legacy fallback discarded configured tile dimensions or overlap");
    }
    for (bool video : {false, true}) {
        RecordingRetryVAE vae(backend);
        vae.mode = RetryVAE::RESERVE_FAILURE;
        sd_tiling_params_t tiling{};
        require(run(vae, tiling, false, video).empty() && vae.attempts.size() == (video ? 1 : 2),
                "auto-fit-off video policy or image fallback bound changed");
    }
    {
        RecordingRetryVAE vae(backend);
        sd_tiling_params_t tiling{};
        require(!run(vae, tiling, true, false).empty() && vae.attempts.size() == 1 && !tiling.enabled,
                "successful decode retried");
    }
}

int main() {
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(backend != nullptr, "CPU backend unavailable");
        test_retry_loop(backend.get());
        RetryVAE vae(backend.get());
        sd::Tensor<float> input({8, 8, 3, 1});
        sd_tiling_params_t tiling{};
        auto run = [&] { vae.calls = 0; return vae.decode(1, input, tiling, false, false, false, true); };
        require(!run().empty() && vae.last_decode_outcome() == VAEDecodeOutcome::SUCCESS && !vae.can_retry_decode(), "finite decode classification");
        vae.mode = RetryVAE::RESERVE_FAILURE;
        require(run().empty() && vae.last_decode_outcome() == VAEDecodeOutcome::ALLOCATION_FAILED && vae.can_retry_decode(), "allocation must allow recovery");
        vae.free_compute_buffer();
        require(vae.can_retry_decode(), "cleanup erased retry eligibility");
        vae.mode = RetryVAE::EMPTY_WITHOUT_COMPUTE;
        require(run().empty() && vae.last_decode_outcome() == VAEDecodeOutcome::FAILED && !vae.can_retry_decode(), "stale allocation status enabled retry");
        vae.mode = RetryVAE::BAD_READBACK;
        require(run().empty() && !vae.can_retry_decode(), "invalid graph result enabled retry");
        for (auto mode : {RetryVAE::NAN_OUTPUT, RetryVAE::INF_OUTPUT}) {
            vae.mode = mode;
            require(run().empty() && vae.last_decode_outcome() == VAEDecodeOutcome::NONFINITE && vae.can_retry_decode(), "nonfinite recovery lost");
        }
        vae.cancel = true;
        require(!vae.can_retry_decode(), "late cancellation did not veto nonfinite recovery");
        require(run().empty() && vae.calls == 0 && vae.last_decode_outcome() == VAEDecodeOutcome::CANCELLED, "pre-decode cancellation dispatched work");
        vae.cancel = false;
        vae.mode = RetryVAE::FINITE;
        vae.cancel_after_compute = true;
        require(run().empty() && vae.last_decode_outcome() == VAEDecodeOutcome::CANCELLED && !vae.can_retry_decode(), "post-compute cancellation enabled recovery");
        vae.cancel_after_compute = false;
        vae.cancel = false;
        require(!run().empty() && vae.last_decode_outcome() == VAEDecodeOutcome::SUCCESS, "subsequent success retained failure");
        // Successful first tile must not conceal failure of the second tile.
        vae.mode = RetryVAE::RESERVE_FAILURE;
        vae.fail_on_call = 2;
        tiling.enabled = true;
        tiling.tile_size_x = tiling.tile_size_y = 4;
        require(run().empty() && vae.calls == 2 && vae.can_retry_decode(), "spatial tile failure lost or processing continued");
        tiling.enabled = false;
        tiling.temporal_tiling = true;
        input = sd::Tensor<float>({4, 4, 8, 3, 1});
        require(run().empty() && vae.calls == 2 && vae.last_decode_outcome() == VAEDecodeOutcome::ALLOCATION_FAILED, "temporal tile failure lost or processing continued");
        vae.mode = RetryVAE::FINITE;
        require(!run().empty() && !vae.can_retry_decode(), "temporal retry recovery failed");
        std::cout << "PASS production VAE retry loop bounds/order/cancellation and actual CPU VAE allocation/readback/nonfinite/cancellation outcomes, stale status reset, spatial and temporal first-failure retention and recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
