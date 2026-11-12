/* Model-free admission regression: real services, dispatcher, model index,
   options and task state. Inference/filter bodies are inert controlled gates;
   no native backend, driver, model loader or image decoder is linked. */
#include "application_services.h"
#include "config.hpp"
#include "cosmo-webgpu.h"
#include "inference_worker.hpp"
#include "llama_service.h"
#include "logging.h"
#include "native_main_executor.hpp"
#include "native_prompt_validation.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>

namespace {
[[noreturn]] void unexpected() {
    std::fputs("Unexpected native backend call in admission fixture\n", stderr);
    std::abort();
}

int image_calls = 0;
std::string last_guide;
int error_resets = 0;

class Gate {
public:
    void block() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        changed_.notify_all();
        assert(changed_.wait_for(lock, std::chrono::seconds(5), [&] { return released_; }));
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(changed_.wait_for(lock, std::chrono::seconds(5), [&] { return entered_; }));
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        changed_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool entered_ = false, released_ = false;
} text_gate, filter_gate;

void expect(const crow::response &response, int code) {
    if (response.code != code) {
        std::fprintf(stderr, "Expected status %d, got %d: %s\n", code, response.code, response.body.c_str());
        std::abort();
    }
}

void check_mutations(ApplicationServices &app, int code) {
    expect(app.request(COSMO_APP_MODELS_REFRESH, "{}"), code);
    expect(app.handleRefreshCheckpoints(), code);
    expect(app.handleRefreshVaeAndTextEncoders(), code);
    expect(app.request(COSMO_APP_CONFIG_SET, "{}"), code);
    expect(app.request(COSMO_APP_SETTINGS_SET, "{}"), code);
    expect(app.request(COSMO_APP_OPTIONS_SET, "{}"), code);
    crow::request empty;
    empty.body = "{}";
    expect(app.handleControlNetDetect(empty), code);
    empty.body = "{\"imageList\":[]}";
    expect(app.handleExtraBatchImages(empty), code);
}
}

void log_message(LogLevel, const char *, ...) {}
void reset_sd_generation_error() { ++error_resets; }
std::string sd_generation_error_message(const std::string &fallback) { return fallback; }
bool cosmo_native_main_required() { return false; }
std::vector<std::string> cosmo_inference_worker(std::function<std::vector<std::string>()> work) {
    return work();
}

ImageGenerator::ImageGenerator(std::shared_ptr<TaskStateManager>, std::shared_ptr<OptionsManager>,
                               std::shared_ptr<SdkitModelIndex>, std::shared_ptr<ImageFilters>,
                               const ServerParams &) {}
ImageGenerator::~ImageGenerator() = default;
bool ImageGenerator::interrupt() { return false; }
std::vector<std::string> ImageGenerator::generateTxt2Img(const ImageGenerationParams &, const std::string &) { ++image_calls; return {}; }
std::vector<std::string> ImageGenerator::generateImg2Img(const ImageGenerationParams &params, const std::string &) { ++image_calls; last_guide = params.inpaint_color_guide_base64; return {}; }
std::vector<std::string> ImageGenerator::generateVideo(const VideoGenerationParams &, const std::string &) { unexpected(); }
ImageFilters::ImageFilters(std::shared_ptr<SdkitModelIndex>) {}
ImageFilters::~ImageFilters() = default;
std::vector<std::string> ImageFilters::upscaleBatch(const std::vector<std::string> &, const std::string &, int) { unexpected(); }
std::vector<std::string> ImageFilters::applyControlNetFilterBatch(const std::vector<std::string> &, const std::string &) {
    filter_gate.block();
    return {};
}

extern "C" {
size_t sd_get_backend_device_count() { unexpected(); }
bool sd_get_backend_device_info(size_t, sd_backend_device_info_t *) { unexpected(); }
sample_method_t str_to_sample_method(const char *) { unexpected(); }
scheduler_t str_to_scheduler(const char *) { unexpected(); }
sdkit_prompt_validation_status sdkit_validate_prompt_attention(const char *, size_t) { unexpected(); }
int cosmo_webgpu_select_device(const char *) { unexpected(); }
const char *cosmo_webgpu_selected_device() { unexpected(); }
const cosmo_webgpu_device_info *cosmo_webgpu_device_metadata(const char *) { unexpected(); }
const char *cosmo_webgpu_default_device() { unexpected(); }
const char *cosmo_webgpu_requested_provider() { unexpected(); }
size_t cosmo_webgpu_unavailable_count() { unexpected(); }
const cosmo_webgpu_unavailable_info *cosmo_webgpu_unavailable_at(size_t) { unexpected(); }
int cosmo_llama_service_generate(const cosmo_llama_request *, cosmo_llama_result *) noexcept {
    text_gate.block();
    return COSMO_LLAMA_OK;
}
void cosmo_llama_result_free(cosmo_llama_result *) noexcept {}
}

int main(int argc, char **argv) {
    assert(cosmo_config_prepare(&argc, &argv) == -1);
    namespace fs = std::filesystem;
    fs::create_directories("checkpoints/nested");
    auto index = std::make_shared<SdkitModelIndex>();
    index->setCheckpointDir(fs::absolute("checkpoints").string());
    ServerParams params;
    params.model_manager = index;
    params.compute_backend = "CPU";
    ApplicationServices app(params);
    assert(!app.busy());

    // Real shared image handler and validator: unsupported independent guides
    // must fail before task creation, error reset or any inference invocation.
    for (const char *field : {"inpaint_color_guide", "inpaint_color_guidance"}) {
        for (const char *value : {"\"data:image/png;base64,aGVsbG8=\"", "\"\"", "false", "0", "[]", "{}"}) {
            crow::request request;
            request.body = std::string("{\"force_task_id\":\"guide-rejected\",\"") + field + "\":" + value + "}";
            const int calls = image_calls, resets = error_resets;
            const auto rejected = app.handleImg2Img(request);
            expect(rejected, 400);
            assert(rejected.body.find(field) != std::string::npos);
            if (std::string(field) == "inpaint_color_guidance")
                assert(rejected.body.find("not supported") != std::string::npos);
            assert(image_calls == calls && error_resets == resets && !app.busy());
            expect(app.request(COSMO_APP_PROGRESS, "{\"id_task\":\"guide-rejected\"}"), 404);
            // These fields remain irrelevant on txt2img, including malformed values.
            request.body = std::string("{\"") + field + "\":" + value + "}";
            expect(app.handleTxt2Img(request), 200);
            assert(image_calls == calls + 1);
        }
        crow::request request;
        request.body = std::string("{\"") + field + "\":null}";
        expect(app.handleImg2Img(request), 200);
    }
    crow::request ordinary;
    ordinary.body = "{\"mask\":\"ordinary-mask\",\"inpaint_color_guide\":null,\"inpaint_color_guidance\":null}";
    expect(app.handleImg2Img(ordinary), 200);
    ordinary.body = "{}";
    expect(app.handleImg2Img(ordinary), 200);
    crow::request advanced;
    advanced.body = R"({"inpaint_color_guide":"RGBA-fixture","init_images":["original-fixture"],"denoising_strength":0.5})";
    expect(app.handleImg2Img(advanced), 200);
    assert(last_guide == "RGBA-fixture");
    for (const char* strength : {"0", "1", "-1", "1.01", "true", "null", "[]", "\"0.5\""}) {
        advanced.body = std::string(R"({"force_task_id":"bad-guide-strength","inpaint_color_guide":"RGBA-fixture","init_images":["original-fixture"],"denoising_strength":)") + strength + "}";
        const int calls = image_calls, resets = error_resets;
        expect(app.handleImg2Img(advanced), 400);
        assert(image_calls == calls && error_resets == resets && !app.busy());
        expect(app.request(COSMO_APP_PROGRESS, R"({"id_task":"bad-guide-strength"})"), 404);
    }
    std::puts("Advanced inpaint guide forwarding/preflight regression passed (inference mocked)");

    // Both the shell and HTTP refresh routes call this exact request method.
    // The old double try-lock fails here, even with an idle empty model index.
    auto refreshed = app.request(COSMO_APP_MODELS_REFRESH, "{}");
    expect(refreshed, 200);
    assert(crow::json::load(refreshed.body)["models"].size() == 0);
    check_mutations(app, 200);

    // Indexing reads filenames/metadata only, never these placeholder bytes.
    std::ofstream("checkpoints/nested/index-only.gguf") << "index fixture, not model weights";
    refreshed = app.request(COSMO_APP_MODELS_REFRESH, "{}");
    expect(refreshed, 200);
    auto listing = crow::json::load(refreshed.body);
    assert(listing["models"].size() == 1);
    assert(listing["models"][0]["name"].s() == "nested/index-only.gguf");
    assert(refreshed.body == app.request(COSMO_APP_MODELS, "{}").body);

    // The real dispatcher and generation guard are held by a controlled,
    // model-free text implementation. Busy mutations must still fail fast.
    std::promise<crow::response> completion;
    auto generated = completion.get_future();
    assert(app.submitGeneration(4, "{\"backend\":\"cpu\"}", [&](crow::response response) {
        completion.set_value(std::move(response));
    }));
    text_gate.wait();
    assert(app.busy());
    assert(!app.submitGeneration(4, "{}", [](crow::response) { unexpected(); }));
    const auto config = cosmo_config_document();
    check_mutations(app, 409);
    assert(cosmo_config_document() == config);
    expect(app.request(COSMO_APP_MODELS, "{}"), 200);
    text_gate.release();
    assert(generated.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    expect(generated.get(), 200);
    assert(!app.busy());
    check_mutations(app, 200);

    // Direct legacy filter work holds only generation_mutex_, not dispatcher
    // admission. Its concurrent mutations must be rejected by the mutex gate.
    auto filter = std::async(std::launch::async, [&] {
        crow::request request;
        request.body = "{\"controlnet_input_images\":[]}";
        return app.handleControlNetDetect(request);
    });
    filter_gate.wait();
    assert(!app.busy());
    check_mutations(app, 409);
    assert(cosmo_config_document() == config);
    expect(app.request(COSMO_APP_TEXT, "{\"backend\":\"cpu\"}"), 409);
    filter_gate.release();
    assert(filter.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    expect(filter.get(), 200);
    check_mutations(app, 200);

    fs::remove("checkpoints/nested/index-only.gguf");
    refreshed = app.request(COSMO_APP_MODELS_REFRESH, "{}");
    expect(refreshed, 200);
    assert(crow::json::load(refreshed.body)["models"].size() == 0);
    app.closeAdmission();
    check_mutations(app, 409);
    std::puts("Application refresh/admission regression passed without native backends");
}
