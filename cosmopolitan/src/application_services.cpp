/* SPDX-License-Identifier: MIT */
#include "application_services.h"
#include "cosmo-webgpu.h"
#include "config.hpp"
#include "inference_worker.hpp"
#include "llama_service.h"
#include "native_main_executor.hpp"
#include "image_utils.h"
#include "logging.h"
#include <algorithm>
#include <atomic>
#include <unistd.h>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Convert webui (Forge/Automatic1111 style) sampler/scheduler names
// into stable-diffusion.cpp compatible names.
static std::string convert_webui_sampler_name(const std::string& name) {
    static const std::unordered_map<std::string, std::string> mapping = {
        {"Euler", "euler"},
        {"Euler a", "euler_a"},
        {"Heun", "heun"},
        {"DPM2", "dpm2"},
        {"DPM++ 2S a", "dpm++2s_a"},
        {"DPM++ 2M", "dpm++2m"},
        {"DPM++ 2M v2", "dpm++2mv2"},
        {"DPM++ 2M SDE", "dpm++2m_sde"},
        {"DPM++ 3M SDE", "dpm++3m_sde"},
        {"UniPC", "unipc"},
        {"DEIS", "deis"},
        {"IPNDM", "ipndm"},
        {"IPNDM_V", "ipndm_v"},
        {"LCM", "lcm"},
        {"RES Multistep", "res_multistep"},
        {"Gradient Estimation", "euler_ge"},
        {"ER-SDE", "er_sde"},
        {"DDIM", "ddim_trailing"},
        {"TCD", "tcd"},
    };

    auto it = mapping.find(name);
    if (it != mapping.end()) return it->second;
    return name;
}

static std::string convert_webui_scheduler_name(const std::string& name) {
    static const std::unordered_map<std::string, std::string> mapping = {
        {"automatic", "discrete"},
        {"uniform", "discrete"},
        {"normal", "discrete"},
        {"karras", "karras"},
        {"exponential", "exponential"},
        {"sgm_uniform", "sgm_uniform"},
        {"simple", "simple"},
        {"beta", "beta"},
        {"kl_optimal", "kl_optimal"},
        {"ddim_uniform", "ddim_uniform"},
        {"linear_quadratic", "linear_quadratic"},
        {"align_your_steps", "ays"},
        {"align_your_steps_GITS", "gits"},
    };

    auto it = mapping.find(name);
    if (it != mapping.end()) return it->second;
    return name;
}

// Convert webui upscaler names to stable-diffusion.cpp compatible names
static std::string convert_webui_upscaler_name(const std::string& name) {
    static const std::unordered_map<std::string, std::string> mapping = {
        {"R-ESRGAN 4x+", "RealESRGAN_x4plus"},
        {"R-ESRGAN 4x+ Anime6B", "RealESRGAN_x4plus_anime_6B"},
        // Add more mappings as needed
    };

    auto it = mapping.find(name);
    if (it != mapping.end()) return it->second;
    return name;
}

// Convert webui ControlNet model names to sdkit compatible names (remove hash suffix)
static std::string convert_webui_controlnet_model_name(const std::string& name) {
    // Remove the hash suffix like " [a3cd7cd6]" from the end
    size_t bracket_pos = name.find_last_of('[');
    if (bracket_pos != std::string::npos) {
        return name.substr(0, bracket_pos - 1);  // -1 to remove the space before [
    }
    return name;
}

static control_net_type_t parse_control_net_type(const crow::json::rvalue& value) {
    int type = CONTROL_NET_TYPE_CANNY;
    if (value.t() == crow::json::type::String) {
        std::string name = value.s();
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        static const std::unordered_map<std::string, int> types = {
            {"openpose", CONTROL_NET_TYPE_OPENPOSE},
            {"pose", CONTROL_NET_TYPE_OPENPOSE},
            {"depth", CONTROL_NET_TYPE_DEPTH},
            {"softedge", CONTROL_NET_TYPE_HED},
            {"soft-edge", CONTROL_NET_TYPE_HED},
            {"hed", CONTROL_NET_TYPE_HED},
            {"scribble", CONTROL_NET_TYPE_SKETCH},
            {"sketch", CONTROL_NET_TYPE_SKETCH},
            {"canny", CONTROL_NET_TYPE_CANNY},
            {"lineart", CONTROL_NET_TYPE_CANNY},
            {"mlsd", CONTROL_NET_TYPE_MLSD},
            {"normal", CONTROL_NET_TYPE_NORMAL},
            {"normalbae", CONTROL_NET_TYPE_NORMAL},
            {"segment", CONTROL_NET_TYPE_SEGMENT},
            {"segmentation", CONTROL_NET_TYPE_SEGMENT},
            {"seg", CONTROL_NET_TYPE_SEGMENT},
            {"content", CONTROL_NET_TYPE_GLOBAL},
            {"global", CONTROL_NET_TYPE_GLOBAL},
        };
        auto it = types.find(name);
        if (it == types.end()) {
            throw std::invalid_argument("Unknown ControlNet condition type: " + name);
        }
        type = it->second;
    } else {
        type = value.i();
    }
    if (type < 0 || type >= CONTROL_NET_TYPE_COUNT) {
        throw std::invalid_argument("Unknown ControlNet condition type id");
    }
    return static_cast<control_net_type_t>(type);
}

static sd_cache_mode_t parse_cache_mode(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (name.empty() || name == "disabled" || name == "none" || name == "off") {
        return SD_CACHE_DISABLED;
    }
    if (name == "easycache" || name == "easy") {
        return SD_CACHE_EASYCACHE;
    }
    if (name == "teacache" || name == "tea") {
        return SD_CACHE_TEACACHE;
    }
    throw std::invalid_argument("Unknown video cache mode: " + name);
}

// Round dimension to nearest multiple of 64
static int round_to_nearest_multiple_of_64(int dimension) {
    return std::round(static_cast<double>(dimension) / 64.0) * 64;
}

static crow::response request_error(int status, const std::string& message) {
    crow::json::wvalue error;
    error["message"] = message;
    return crow::response(status, error);
}

class NativeRequestConflict : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

class ApplicationServices::GenerationRequestGuard {
   public:
    explicit GenerationRequestGuard(ApplicationServices& server)
        : server_(server), lock_(server.generation_mutex_, std::try_to_lock) {}

    bool acquired() const { return lock_.owns_lock(); }
    const std::string& taskId() const { return task_id_; }

    void start(const crow::json::rvalue& body, const char* kind) {
        if (body.t() != crow::json::type::Object) {
            throw std::invalid_argument("Generation request must be an object");
        }
        if (body.has("force_task_id")) {
            if (body["force_task_id"].t() != crow::json::type::String) {
                throw std::invalid_argument("force_task_id must be a string");
            }
            task_id_ = std::string(body["force_task_id"].s());
            if (task_id_.empty() || task_id_.size() > 128 || task_id_.find('\0') != std::string::npos) {
                throw std::invalid_argument("force_task_id must contain 1..128 bytes without NUL");
            }
            if (server_.task_state_manager_->taskExists(task_id_)) {
                throw NativeRequestConflict("force_task_id has already been used");
            }
        } else {
            do {
                task_id_ = std::string("native-") + kind + "-" + std::to_string(++server_.request_sequence_);
            } while (server_.task_state_manager_->taskExists(task_id_));
        }

        const auto overrides = body.has("override_settings")
                                   ? body["override_settings"]
                                   : crow::json::load("{}");
        if (overrides.t() != crow::json::type::Object) {
            throw std::invalid_argument("override_settings must be an object");
        }
        if (overrides.has("sd_model_checkpoint")) {
            server_.validateCheckpoint(overrides["sd_model_checkpoint"]);
        }
        server_.options_manager_->beginRequestOverrides(overrides);
        overrides_active_ = true;
        server_.task_state_manager_->createTask(task_id_);
        task_created_ = true;
        std::lock_guard<std::mutex> state_lock(server_.active_task_mutex_);
        server_.active_task_id_ = task_id_;
        server_.active_task_kind_ = kind;
        active_ = true;
    }

    ~GenerationRequestGuard() noexcept {
        try {
            if (active_) {
                std::lock_guard<std::mutex> state_lock(server_.active_task_mutex_);
                server_.active_task_id_.clear();
                server_.active_task_kind_.clear();
            }
            if (task_created_ &&
                !server_.task_state_manager_->getTaskProgressState(task_id_, false, -1).completed) {
                server_.task_state_manager_->completeTask(
                    task_id_, {}, "{\"message\":\"Generation ended without a result\"}");
            }
        } catch (...) {
            std::fputs("Failed to finalize native generation task\n", stderr);
        }
        try {
            if (overrides_active_) server_.options_manager_->endRequestOverrides();
        } catch (...) {
            std::fputs("Failed to clear native request options\n", stderr);
        }
    }

    GenerationRequestGuard(const GenerationRequestGuard&) = delete;
    GenerationRequestGuard& operator=(const GenerationRequestGuard&) = delete;

   private:
    ApplicationServices& server_;
    std::unique_lock<std::mutex> lock_;
    std::string task_id_;
    bool overrides_active_ = false;
    bool task_created_ = false;
    bool active_ = false;
};

void ApplicationServices::validateCheckpoint(const crow::json::rvalue& value, bool allow_empty) {
    if (value.t() != crow::json::type::String) {
        throw std::invalid_argument("sd_model_checkpoint must be an indexed checkpoint name");
    }
    const std::string name = value.s();
    if (allow_empty && name.empty()) return;
    const auto names = model_manager_->getModelNamesByType(ModelType::CHECKPOINT);
    if (name.empty() || std::find(names.begin(), names.end(), name) == names.end()) {
        throw std::invalid_argument("Checkpoint is not in the native model index");
    }
}


namespace {
std::atomic<bool> application_live{false};
void *acquire_application() {
    if (application_live.exchange(true))
        throw std::runtime_error("Only one live application service is allowed per process");
    return &application_live;
}
void release_application(void *) { application_live.store(false); }
}

ApplicationServices::ApplicationServices(const ServerParams &params)
    : process_lease_(acquire_application(), release_application),
      params_(params), model_manager_(params.model_manager) {
    static std::atomic<unsigned long> instances{0};
    instance_id_ = "application-" + std::to_string(getpid()) + "-" + std::to_string(++instances);
    if (cosmo_native_main_required()) {
        const size_t count = sd_get_backend_device_count();
        for (size_t i = 0; i < count; ++i) {
            sd_backend_device_info_t info{};
            if (!sd_get_backend_device_info(i, &info))
                throw std::runtime_error("Cannot initialize native backend metadata");
        }
    }
    if (!model_manager_) throw std::invalid_argument("Application requires a model index");
    options_manager_ = std::make_shared<OptionsManager>();
    task_state_manager_ = std::make_shared<TaskStateManager>();
    image_filters_ = std::make_shared<ImageFilters>(model_manager_);
    image_generator_ = std::make_unique<ImageGenerator>(task_state_manager_, options_manager_,
                                                       model_manager_, image_filters_, params);
    if (!options_manager_->load()) throw std::runtime_error("Cannot load persistent generation options");
}
ApplicationServices::~ApplicationServices() { shutdown(); }
void ApplicationServices::closeAdmission() {
    admission_closed_.store(true);
    inference_dispatcher_.close_admission();
}
void ApplicationServices::interrupt() {
    std::lock_guard<std::mutex> lock(active_task_mutex_);
    if (!active_task_id_.empty() && active_task_kind_ == "text")
        task_state_manager_->interruptTask(active_task_id_);
    else image_generator_->interrupt();
}
void ApplicationServices::shutdown() {
    closeAdmission();
    interrupt();
    inference_dispatcher_.shutdown();
}

crow::response ApplicationServices::handlePing() { return crow::response(200, "OK"); }

static std::string cosmo_backend_token(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

static std::string cosmo_request_backend(const std::string &spec) {
    if (spec.empty() || spec.size() > 4096) throw std::invalid_argument("backend must name available devices");
    // Populate the registry before resolving process-local selectors or genuine stable IDs.
    (void)sd_get_backend_device_count();
    std::istringstream assignments(spec);
    std::string assignment, result;
    while (std::getline(assignments, assignment, ',')) {
        assignment = cosmo_backend_token(assignment);
        const auto equals = assignment.find('=');
        std::string prefix;
        if (equals != std::string::npos) {
            const std::string module = cosmo_backend_token(assignment.substr(0, equals));
            if (module.empty() || module.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.*-") != std::string::npos)
                throw std::invalid_argument("Invalid backend module assignment");
            prefix = module + "=";
            assignment = assignment.substr(equals + 1);
        }
        std::istringstream devices(assignment);
        std::string token, resolved;
        while (std::getline(devices, token, '&')) {
            token = cosmo_backend_token(token);
            if (token.empty()) throw std::invalid_argument("Empty backend device selector");
            std::string lower = token;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return std::tolower(ch); });
            std::string selected;
            if (lower == "cpu") {
                selected = "CPU";
            } else {
                const char *request = lower == "webgpu" || lower == "auto" ? "auto" : token.c_str();
                if (cosmo_webgpu_select_device(request))
                    throw std::invalid_argument("Requested WebGPU device is unavailable: " + token);
                const auto *info = cosmo_webgpu_device_metadata("auto");
                if (!info) throw std::invalid_argument("Requested WebGPU device has no metadata");
                selected = info->selector;
            }
            if (!resolved.empty()) resolved += "&";
            resolved += selected;
        }
        if (resolved.empty() || (!assignment.empty() && assignment.back() == '&'))
            throw std::invalid_argument("Empty backend device selector");
        if (!result.empty()) result += ",";
        result += prefix + resolved;
    }
    if (result.empty() || spec.back() == ',') throw std::invalid_argument("Empty backend assignment");
    return result;
}

crow::response ApplicationServices::handleBackendDevices() {
    crow::json::wvalue::list devices;
    for (size_t index = 0; index < sd_get_backend_device_count(); ++index) {
        sd_backend_device_info_t info{};
        if (!sd_get_backend_device_info(index, &info)) continue;
        const std::string selector = info.name ? info.name : "";
        const auto *native = cosmo_webgpu_device_metadata(selector.c_str());
        crow::json::wvalue device;
        device["selector"] = selector;
        device["backend"] = std::string(info.backend ? info.backend : "");
        device["description"] = std::string(info.description ? info.description : "");
        device["vendor"] = std::string(info.vendor ? info.vendor : "");
        device["provider"] = native ? native->provider : "builtin";
        device["software"] = native ? native->software != 0 : info.type == SD_BACKEND_DEVICE_TYPE_CPU;
        device["stable_id"] = native && native->stable_id ? native->stable_id : "";
        device["stable_id_available"] = native && native->stable_id && *native->stable_id;
        device["selector_scope"] = "process";
        const bool memory_known = native ? native->memory_known != 0 : info.memory_total != 0;
        device["memory_known"] = memory_known;
        if (memory_known) {
            device["memory_free"] = static_cast<uint64_t>(native ? native->memory_free : info.memory_free);
            device["memory_total"] = static_cast<uint64_t>(native ? native->memory_total : info.memory_total);
        } else {
            device["memory_free"] = nullptr;
            device["memory_total"] = nullptr;
        }
        if (native && native->software) {
            device["type"] = "software";
        } else {
            switch (info.type) {
                case SD_BACKEND_DEVICE_TYPE_CPU: device["type"] = "cpu"; break;
                case SD_BACKEND_DEVICE_TYPE_IGPU: device["type"] = "integrated-gpu"; break;
                case SD_BACKEND_DEVICE_TYPE_GPU: device["type"] = "gpu"; break;
                default: device["type"] = "unknown"; break;
            }
        }
        devices.emplace_back(std::move(device));
    }
    crow::json::wvalue response;
    response["devices"] = std::move(devices);
    response["default_webgpu_selector"] = cosmo_webgpu_default_device();
    response["provider_policy"] = cosmo_webgpu_requested_provider();
    crow::json::wvalue::list unavailable;
    for (size_t index = 0; index < cosmo_webgpu_unavailable_count(); ++index) {
        const auto *entry = cosmo_webgpu_unavailable_at(index);
        if (!entry) continue;
        crow::json::wvalue item;
        item["provider"] = entry->provider;
        item["name"] = entry->name;
        item["adapter_type"] = entry->adapter_type;
        item["reason"] = entry->reason;
        unavailable.emplace_back(std::move(item));
    }
    response["unavailable"] = std::move(unavailable);
    return crow::response(200, response);
}

crow::response ApplicationServices::handleGetOptions() {
    try {
        auto options = options_manager_->getOptions(false);
        return crow::response(200, options);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to get options: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handlePostOptions(const crow::request& req) {
    if (inference_dispatcher_.busy())
        return request_error(409, "A native generation request is active");
    std::unique_lock<std::mutex> generation_lock(generation_mutex_, std::try_to_lock);
    if (!generation_lock.owns_lock()) {
        return request_error(409, "A native generation request is active");
    }
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body || json_body.t() != crow::json::type::Object) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }

        if (json_body.has("sd_model_checkpoint")) {
            validateCheckpoint(json_body["sd_model_checkpoint"], true);
        }
        if (options_manager_->setOptions(json_body)) {
            return crow::response(200, "OK");
        } else {
            crow::json::wvalue error;
            error["message"] = "Failed to save options";
            return crow::response(500, error);
        }
    } catch (const std::invalid_argument& e) {
        return request_error(400, e.what());
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to set options: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleCheckpoints() {
    try {
        crow::json::wvalue::list models;
        for (const auto& name : model_manager_->getModelNamesByType(ModelType::CHECKPOINT)) {
            crow::json::wvalue model;
            model["name"] = name;
            models.emplace_back(std::move(model));
        }
        crow::json::wvalue response;
        response["models"] = std::move(models);
        return crow::response(200, response);
    } catch (const std::exception& e) {
        return request_error(500, std::string("Failed to list checkpoints: ") + e.what());
    }
}

crow::response ApplicationServices::handleTxt2Img(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }

        return generateImage(json_body, false);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to generate image: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleImg2Img(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }

        return generateImage(json_body, true);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to generate image: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleTxt2Video(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }
        return generateVideo(json_body, false);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to generate video: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleImg2Video(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }
        return generateVideo(json_body, true);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to generate video: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::generateVideo(const crow::json::rvalue& json_body, bool is_img2video) {
    GenerationRequestGuard request(*this);
    if (!request.acquired()) return request_error(409, "A native generation request is active");
    std::string task_id;
    try {
        request.start(json_body, "video");
        task_id = request.taskId();
        reset_sd_generation_error();
        VideoGenerationParams params;
        if (json_body.has("audio_vae_path")) {
            if (json_body["audio_vae_path"].t() != crow::json::type::String) {
                throw std::invalid_argument("audio_vae_path must be a model path string");
            }
            params.audio_vae_path = std::string(json_body["audio_vae_path"].s());
        }
        if (json_body.has("embeddings_connectors_path")) {
            if (json_body["embeddings_connectors_path"].t() != crow::json::type::String) {
                throw std::invalid_argument("embeddings_connectors_path must be a model path string");
            }
            params.embeddings_connectors_path = std::string(json_body["embeddings_connectors_path"].s());
        }
        if (json_body.has("backend")) {
            if (json_body["backend"].t() != crow::json::type::String) {
                throw std::invalid_argument("backend must be a device-assignment string");
            }
            params.backend = cosmo_request_backend(std::string(json_body["backend"].s()));
        }
        params.prompt          = json_body.has("prompt") ? std::string(json_body["prompt"].s()) : "";
        params.negative_prompt = json_body.has("negative_prompt")
                                     ? std::string(json_body["negative_prompt"].s())
                                     : "";
        params.width     = json_body.has("width") ? json_body["width"].i() : 512;
        params.height    = json_body.has("height") ? json_body["height"].i() : 512;
        params.steps     = json_body.has("steps") ? json_body["steps"].i() : 20;
        params.frames    = json_body.has("video_frames") ? json_body["video_frames"].i() : 25;
        params.fps       = json_body.has("fps") ? json_body["fps"].i() : 8;
        params.motion_bucket_id = json_body.has("motion_bucket_id")
                                      ? json_body["motion_bucket_id"].i()
                                      : 127;
        params.augmentation_level = json_body.has("augmentation_level")
                                        ? static_cast<float>(json_body["augmentation_level"].d())
                                        : 0.0f;
        params.cfg_scale = json_body.has("cfg_scale") ? static_cast<float>(json_body["cfg_scale"].d()) : 5.0f;
        params.strength  = json_body.has("denoising_strength")
                               ? static_cast<float>(json_body["denoising_strength"].d())
                               : 0.75f;
        params.seed = json_body.has("seed") ? static_cast<int64_t>(json_body["seed"].i()) : -1;
        if (json_body.has("clip_skip")) {
            params.clip_skip = json_body["clip_skip"].i();
        }
        if (json_body.has("flow_shift")) {
            params.flow_shift = static_cast<float>(json_body["flow_shift"].d());
        }
        if (json_body.has("moe_boundary")) {
            params.moe_boundary = static_cast<float>(json_body["moe_boundary"].d());
        }

        if (json_body.has("sampler_name") && json_body["sampler_name"].t() == crow::json::type::String) {
            const std::string sampler = convert_webui_sampler_name(std::string(json_body["sampler_name"].s()));
            params.sampler = str_to_sample_method(sampler.c_str());
        }
        if (json_body.has("scheduler") && json_body["scheduler"].t() == crow::json::type::String) {
            const std::string scheduler = convert_webui_scheduler_name(std::string(json_body["scheduler"].s()));
            params.scheduler = str_to_scheduler(scheduler.c_str());
        }
        if (json_body.has("lora_paths") && json_body["lora_paths"].t() == crow::json::type::List) {
            for (size_t i = 0; i < json_body["lora_paths"].size(); ++i) {
                params.lora_paths.emplace_back(json_body["lora_paths"][i].s());
            }
        }
        if (json_body.has("lora_alphas") && json_body["lora_alphas"].t() == crow::json::type::List) {
            for (size_t i = 0; i < json_body["lora_alphas"].size(); ++i) {
                params.lora_alphas.push_back(static_cast<float>(json_body["lora_alphas"][i].d()));
            }
        }

        if (json_body.has("init_images") && json_body["init_images"].t() == crow::json::type::List &&
            json_body["init_images"].size() > 0) {
            params.init_image_base64 = std::string(json_body["init_images"][0].s());
        } else if (json_body.has("init_image") && json_body["init_image"].t() == crow::json::type::String) {
            params.init_image_base64 = std::string(json_body["init_image"].s());
        }
        if (json_body.has("end_image") && json_body["end_image"].t() == crow::json::type::String) {
            params.end_image_base64 = std::string(json_body["end_image"].s());
        }
        if (is_img2video && params.init_image_base64.empty()) {
            throw std::invalid_argument("img2video requires an initial image");
        }

        if (json_body.has("cache") && json_body["cache"].t() == crow::json::type::Object) {
            const auto cache = json_body["cache"];
            if (cache.has("mode") && cache["mode"].t() == crow::json::type::String) {
                params.cache_mode = parse_cache_mode(std::string(cache["mode"].s()));
            }
            if (cache.has("threshold")) {
                params.cache_threshold = static_cast<float>(cache["threshold"].d());
            }
            auto percent = [](double value) {
                return static_cast<float>(value > 1.0 ? value * 0.01 : value);
            };
            if (cache.has("start_percent")) {
                params.cache_start_percent = percent(cache["start_percent"].d());
            }
            if (cache.has("end_percent")) {
                params.cache_end_percent = percent(cache["end_percent"].d());
            }
        }

        params.width  -= params.width % 8;
        params.height -= params.height % 8;
        if (params.width < 64 || params.height < 64 || params.width > 2048 || params.height > 2048) {
            throw std::invalid_argument("Video width and height must be between 64 and 2048 pixels");
        }
        if (params.steps < 1 || params.steps > 200) {
            throw std::invalid_argument("Video steps must be between 1 and 200");
        }
        if (params.frames < 1 || params.frames > 513) {
            throw std::invalid_argument("video_frames must be between 1 and 513");
        }
        if (params.fps < 1 || params.fps > 60) {
            throw std::invalid_argument("fps must be between 1 and 60");
        }
        if (params.motion_bucket_id < 0 || params.motion_bucket_id > 1023) {
            throw std::invalid_argument("motion_bucket_id must be between 0 and 1023");
        }
        if (!std::isfinite(params.augmentation_level) || params.augmentation_level < 0.0f) {
            throw std::invalid_argument("augmentation_level must be a finite non-negative number");
        }
        if (params.cache_start_percent < 0.0f || params.cache_end_percent > 1.0f ||
            params.cache_start_percent >= params.cache_end_percent) {
            throw std::invalid_argument("Cache range must satisfy 0 <= start < end <= 100 percent");
        }
        if (std::isfinite(params.cache_threshold) && params.cache_threshold < 0.0f) {
            throw std::invalid_argument("Cache threshold cannot be negative");
        }

        std::vector<std::string> frames = cosmo_inference_worker([&] {
            return image_generator_->generateVideo(params, task_id);
        });
        crow::json::wvalue info;
        info["prompt"]       = params.prompt;
        info["seed"]         = params.seed;
        info["width"]        = params.width;
        info["height"]       = params.height;
        info["fps"]          = params.fps;
        info["video_frames"] = static_cast<int>(frames.size());
        info["cache_mode"]   = params.cache_mode == SD_CACHE_EASYCACHE
                                    ? "easycache"
                                    : (params.cache_mode == SD_CACHE_TEACACHE ? "teacache" : "disabled");

        // Do not retain another base64 copy of every frame in TaskState; the
        // response already owns them and video sequences are large.
        task_state_manager_->completeTask(task_id, {}, info.dump());
        crow::json::wvalue response;
        response["frames"] = frames;
        response["task_id"] = task_id;
        response["fps"]    = params.fps;
        response["info"]   = std::move(info);
        return crow::response(200, response);
    } catch (const NativeRequestConflict& e) {
        return request_error(409, e.what());
    } catch (const std::invalid_argument& e) {
        return request_error(400, e.what());
    } catch (const std::exception& e) {
        const std::string message = sd_generation_error_message(std::string("Video generation failed: ") + e.what());
        LOG_ERROR("%s", message.c_str());
        crow::json::wvalue error;
        error["message"] = message;
        task_state_manager_->completeTask(task_id, {}, error.dump());
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::generateImage(const crow::json::rvalue& json_body, bool is_img2img) {
    GenerationRequestGuard request(*this);
    if (!request.acquired()) return request_error(409, "A native generation request is active");
    std::string task_id;
    try {
        request.start(json_body, "image");
        task_id = request.taskId();
        reset_sd_generation_error();
        // Parse generation parameters
        ImageGenerationParams params;
        if (json_body.has("backend")) {
            if (json_body["backend"].t() != crow::json::type::String) {
                throw std::invalid_argument("backend must be a device-assignment string");
            }
            params.backend = cosmo_request_backend(std::string(json_body["backend"].s()));
        }
        params.prompt = json_body.has("prompt") ? std::string(json_body["prompt"].s()) : "";
        params.negative_prompt = json_body.has("negative_prompt") ? std::string(json_body["negative_prompt"].s()) : "";
        if (json_body.has("lora_paths") && json_body["lora_paths"].t() == crow::json::type::List) {
            for (size_t i = 0; i < json_body["lora_paths"].size(); i++) {
                params.lora_paths.push_back(std::string(json_body["lora_paths"][i].s()));
            }
        }
        if (json_body.has("lora_alphas") && json_body["lora_alphas"].t() == crow::json::type::List) {
            for (size_t i = 0; i < json_body["lora_alphas"].size(); i++) {
                params.lora_alphas.push_back(static_cast<float>(json_body["lora_alphas"][i].d()));
            }
        }
        params.width = json_body.has("width") ? json_body["width"].i() : 512;
        params.height = json_body.has("height") ? json_body["height"].i() : 512;
        params.steps = json_body.has("steps") ? json_body["steps"].i() : 20;
        params.cfg_scale = json_body.has("cfg_scale") ? json_body["cfg_scale"].d() : 7.0f;
        params.seed = json_body.has("seed") ? json_body["seed"].i() : -1;
        params.batch_count = json_body.has("batch_size") ? json_body["batch_size"].i() : 1;

        // Sampler and scheduler parameters
        if (json_body.has("sampler_name") && json_body["sampler_name"].t() == crow::json::type::String) {
            std::string sampler_str = json_body["sampler_name"].s();
            // Convert from webui-style sampler name to sd.cpp name
            sampler_str = convert_webui_sampler_name(sampler_str);
            params.sampler = str_to_sample_method(sampler_str.c_str());
        }
        if (json_body.has("scheduler") && json_body["scheduler"].t() == crow::json::type::String) {
            std::string scheduler_str = json_body["scheduler"].s();
            // Convert from webui-style scheduler name to sd.cpp name
            scheduler_str = convert_webui_scheduler_name(scheduler_str);
            params.scheduler = str_to_scheduler(scheduler_str.c_str());
        }

        // img2img specific parameters
        if (is_img2img) {
            if (json_body.has("init_images") && json_body["init_images"].size() > 0) {
                params.init_image_base64 = std::string(json_body["init_images"][0].s());
            }
            if (json_body.has("mask")) {
                params.mask_base64 = std::string(json_body["mask"].s());
            }
            params.strength = json_body.has("denoising_strength") ? json_body["denoising_strength"].d() : 0.75f;
        }

        // reference images
        if (json_body.has("ref_images") && json_body["ref_images"].size() > 0) {
            for (size_t i = 0; i < json_body["ref_images"].size(); i++) {
                params.ref_images_base64.push_back(std::string(json_body["ref_images"][i].s()));
            }
        }

        // ControlNet parameters from alwayson_scripts
        if (json_body.has("alwayson_scripts") && json_body["alwayson_scripts"].has("controlnet")) {
            auto controlnet_obj = json_body["alwayson_scripts"]["controlnet"];
            if (controlnet_obj.has("args") && controlnet_obj["args"].size() > 0) {
                auto controlnet_args = controlnet_obj["args"][0];

                // Extract control image
                if (controlnet_args.has("image")) {
                    params.control_image_base64 = std::string(controlnet_args["image"].s());
                }

                // Extract control strength (weight)
                if (controlnet_args.has("weight")) {
                    params.control_strength = controlnet_args["weight"].d();
                }

                if (controlnet_args.has("union_control_type")) {
                    params.control_net_type = parse_control_net_type(controlnet_args["union_control_type"]);
                } else if (controlnet_args.has("control_type")) {
                    params.control_net_type = parse_control_net_type(controlnet_args["control_type"]);
                }

                // Extract controlnet model name
                if (controlnet_args.has("model")) {
                    std::string webui_model_name = std::string(controlnet_args["model"].s());
                    params.controlnet_model = convert_webui_controlnet_model_name(webui_model_name);
                }

                LOG_INFO("ControlNet params: model='%s', strength=%.2f, Union type=%d, has_image=%s",
                         params.controlnet_model.c_str(),
                         params.control_strength,
                         static_cast<int>(params.control_net_type),
                         params.control_image_base64.empty() ? "no" : "yes");
            }
        }

        // Native sdkit3 ControlNet-LLLite payload. The model path is resolved by
        // Easy Diffusion before this request reaches the backend.
        if (json_body.has("controlnet_lllite") &&
            json_body["controlnet_lllite"].t() == crow::json::type::Object) {
            const auto lllite = json_body["controlnet_lllite"];
            if (lllite.has("model_path") && lllite["model_path"].t() == crow::json::type::String) {
                params.control_net_lllite_model_path = std::string(lllite["model_path"].s());
            }
            if (lllite.has("image") && lllite["image"].t() == crow::json::type::String) {
                params.control_net_lllite_image_base64 = std::string(lllite["image"].s());
            }
            if (lllite.has("strength")) {
                params.control_net_lllite_strength = static_cast<float>(lllite["strength"].d());
            }
            if (lllite.has("start_percent")) {
                params.control_net_lllite_start_percent = static_cast<float>(lllite["start_percent"].d());
            }
            if (lllite.has("end_percent")) {
                params.control_net_lllite_end_percent = static_cast<float>(lllite["end_percent"].d());
            }
            params.control_net_lllite_strength = std::clamp(params.control_net_lllite_strength, -10.f, 10.f);
            params.control_net_lllite_start_percent = std::clamp(params.control_net_lllite_start_percent, 0.f, 100.f);
            params.control_net_lllite_end_percent = std::clamp(params.control_net_lllite_end_percent, 0.f, 100.f);

            if (params.control_net_lllite_model_path.empty() != params.control_net_lllite_image_base64.empty()) {
                throw std::invalid_argument("ControlNet-LLLite requires both model_path and image");
            }
            LOG_INFO("ControlNet-LLLite params: model='%s', strength=%.2f, range=%.1f%%-%.1f%%, has_image=%s",
                     params.control_net_lllite_model_path.c_str(),
                     params.control_net_lllite_strength,
                     params.control_net_lllite_start_percent,
                     params.control_net_lllite_end_percent <= 0.f ? 100.f : params.control_net_lllite_end_percent,
                     params.control_net_lllite_image_base64.empty() ? "no" : "yes");
        }

        // Native base IP-Adapter payload. Paths are explicit so a render can
        // select a matching SD1.5/SDXL adapter and CLIP vision checkpoint.
        if (json_body.has("ip_adapter") &&
            json_body["ip_adapter"].t() == crow::json::type::Object) {
            const auto adapter = json_body["ip_adapter"];
            if (adapter.has("model_path") && adapter["model_path"].t() == crow::json::type::String) {
                params.ip_adapter_model_path = std::string(adapter["model_path"].s());
            }
            if (adapter.has("clip_vision_path") && adapter["clip_vision_path"].t() == crow::json::type::String) {
                params.ip_adapter_clip_vision_path = std::string(adapter["clip_vision_path"].s());
            }
            if (adapter.has("image") && adapter["image"].t() == crow::json::type::String) {
                params.ip_adapter_image_base64 = std::string(adapter["image"].s());
            }
            if (adapter.has("strength")) {
                params.ip_adapter_strength = static_cast<float>(adapter["strength"].d());
            }
            if (adapter.has("start_percent")) {
                params.ip_adapter_start_percent = static_cast<float>(adapter["start_percent"].d());
            }
            if (adapter.has("end_percent")) {
                params.ip_adapter_end_percent = static_cast<float>(adapter["end_percent"].d());
            }
            params.ip_adapter_strength = std::clamp(params.ip_adapter_strength, -10.f, 10.f);
            params.ip_adapter_start_percent = std::clamp(params.ip_adapter_start_percent, 0.f, 100.f);
            params.ip_adapter_end_percent = std::clamp(params.ip_adapter_end_percent, 0.f, 100.f);
            const bool any_ip_adapter_field = !params.ip_adapter_model_path.empty() ||
                                              !params.ip_adapter_clip_vision_path.empty() ||
                                              !params.ip_adapter_image_base64.empty();
            if (any_ip_adapter_field &&
                (params.ip_adapter_model_path.empty() ||
                 params.ip_adapter_clip_vision_path.empty() ||
                 params.ip_adapter_image_base64.empty())) {
                throw std::invalid_argument("IP-Adapter requires model_path, clip_vision_path, and image");
            }
            if (any_ip_adapter_field && params.ip_adapter_end_percent <= params.ip_adapter_start_percent) {
                throw std::invalid_argument("IP-Adapter end_percent must be greater than start_percent");
            }
            LOG_INFO("IP-Adapter params: model='%s', clip_vision='%s', strength=%.3f, range=%.1f%%-%.1f%%, has_image=%s",
                     params.ip_adapter_model_path.c_str(),
                     params.ip_adapter_clip_vision_path.c_str(),
                     params.ip_adapter_strength,
                     params.ip_adapter_start_percent,
                     params.ip_adapter_end_percent,
                     params.ip_adapter_image_base64.empty() ? "no" : "yes");
        }

        if (json_body.has("latent_interposer") &&
            json_body["latent_interposer"].t() == crow::json::type::Object) {
            const auto interposer = json_body["latent_interposer"];
            params.latent_interposer_enabled = !interposer.has("enabled") || interposer["enabled"].b();
            std::string source = "fx";
            if (interposer.has("source") && interposer["source"].t() == crow::json::type::String) {
                source = std::string(interposer["source"].s());
            }
            if (source == "v1") {
                params.latent_interposer_source = SD_LATENT_SOURCE_V1;
            } else if (source == "xl") {
                params.latent_interposer_source = SD_LATENT_SOURCE_XL;
            } else if (source == "v3") {
                params.latent_interposer_source = SD_LATENT_SOURCE_V3;
            } else if (source == "fx") {
                params.latent_interposer_source = SD_LATENT_SOURCE_FX;
            } else {
                throw std::invalid_argument("Latent Interposer source must be one of v1, xl, v3, or fx");
            }
            if (interposer.has("model_path") && interposer["model_path"].t() == crow::json::type::String) {
                params.latent_interposer_model_path = std::string(interposer["model_path"].s());
            }
            if (interposer.has("furception_vae_path") && interposer["furception_vae_path"].t() == crow::json::type::String) {
                params.furception_vae_path = std::string(interposer["furception_vae_path"].s());
            }
            if (interposer.has("source_image") && interposer["source_image"].t() == crow::json::type::String) {
                params.latent_interposer_source_image_base64 = std::string(interposer["source_image"].s());
            }
            if (interposer.has("phase_x")) {
                params.latent_interposer_phase_x = static_cast<int>(interposer["phase_x"].i());
            }
            if (interposer.has("phase_y")) {
                params.latent_interposer_phase_y = static_cast<int>(interposer["phase_y"].i());
            }
            params.latent_interposer_phase_x = std::clamp(params.latent_interposer_phase_x, -100000, 100000);
            params.latent_interposer_phase_y = std::clamp(params.latent_interposer_phase_y, -100000, 100000);
            if (params.latent_interposer_enabled && params.latent_interposer_source == SD_LATENT_SOURCE_V1 &&
                (params.furception_vae_path.empty() || params.latent_interposer_source_image_base64.empty())) {
                throw std::invalid_argument("Furception v1 source requires furception_vae_path and source_image");
            }
            if (!params.latent_interposer_enabled) {
                params.latent_interposer_model_path.clear();
                params.furception_vae_path.clear();
                params.latent_interposer_source_image_base64.clear();
            }
            LOG_INFO("Latent Interposer params: enabled=%s, source='%s', model='%s', phase=%d,%d",
                     params.latent_interposer_enabled ? "true" : "false",
                     source.c_str(),
                     params.latent_interposer_model_path.c_str(),
                     params.latent_interposer_phase_x,
                     params.latent_interposer_phase_y);
        }

        if (json_body.has("vae_interposer") &&
            json_body["vae_interposer"].t() == crow::json::type::Object) {
            const auto bridge = json_body["vae_interposer"];
            const bool enabled = !bridge.has("enabled") || bridge["enabled"].b();
            std::string vae_family;
            std::string model_family;
            if (bridge.has("vae_family") && bridge["vae_family"].t() == crow::json::type::String) {
                vae_family = std::string(bridge["vae_family"].s());
            }
            if (bridge.has("model_family") && bridge["model_family"].t() == crow::json::type::String) {
                model_family = std::string(bridge["model_family"].s());
            }
            if (vae_family == "v1") {
                params.latent_interposer_vae_format = SD_VAE_FORMAT_V1;
            } else if (vae_family == "xl") {
                params.latent_interposer_vae_format = SD_VAE_FORMAT_XL;
            } else if (vae_family == "v3") {
                params.latent_interposer_vae_format = SD_VAE_FORMAT_SD3;
            } else if (vae_family == "fx") {
                params.latent_interposer_vae_format = SD_VAE_FORMAT_FLUX;
            } else {
                throw std::invalid_argument("VAE Interposer family must be one of v1, xl, v3, or fx");
            }
            if (bridge.has("encode_model_path") && bridge["encode_model_path"].t() == crow::json::type::String) {
                params.latent_interposer_encode_model_path = std::string(bridge["encode_model_path"].s());
            }
            if (bridge.has("decode_model_path") && bridge["decode_model_path"].t() == crow::json::type::String) {
                params.latent_interposer_decode_model_path = std::string(bridge["decode_model_path"].s());
            }
            if (!enabled) {
                params.latent_interposer_encode_model_path.clear();
                params.latent_interposer_decode_model_path.clear();
                params.latent_interposer_vae_format = SD_VAE_FORMAT_AUTO;
            } else if (params.latent_interposer_encode_model_path.empty() &&
                       params.latent_interposer_decode_model_path.empty()) {
                throw std::invalid_argument("VAE Interposer requires an encode_model_path or decode_model_path");
            }
            LOG_INFO("VAE Interposer params: VAE='%s', model='%s', encode='%s', decode='%s'",
                     vae_family.c_str(),
                     model_family.c_str(),
                     params.latent_interposer_encode_model_path.c_str(),
                     params.latent_interposer_decode_model_path.c_str());
        }

        // Round dimensions to nearest multiple of 64 when ControlNet is used
        if (!params.controlnet_model.empty() || !params.control_net_lllite_model_path.empty()) {
            int original_width = params.width;
            int original_height = params.height;
            params.width = round_to_nearest_multiple_of_64(params.width);
            params.height = round_to_nearest_multiple_of_64(params.height);
            LOG_INFO("ControlNet detected, rounded dimensions from %dx%d to %dx%d", original_width, original_height,
                     params.width, params.height);
        }

        // Keep this request's gate and parameters alive until the explicitly
        // sized worker completes model loading, generation, and PNG encoding.
        std::vector<std::string> images = cosmo_inference_worker([&] {
            return is_img2img ? image_generator_->generateImg2Img(params, task_id)
                             : image_generator_->generateTxt2Img(params, task_id);
        });

        // Create info JSON string
        crow::json::wvalue info_json;
        info_json["prompt"] = params.prompt;
        info_json["negative_prompt"] = params.negative_prompt;
        info_json["steps"] = params.steps;
        info_json["cfg_scale"] = params.cfg_scale;
        info_json["seed"] = params.seed;
        info_json["width"] = params.width;
        info_json["height"] = params.height;

        crow::json::wvalue infotexts_json;
        infotexts_json["infotexts"] = info_json.dump();
        std::string info = infotexts_json.dump();

        // Complete task
        task_state_manager_->completeTask(task_id, images, info);

        // Return response
        crow::json::wvalue response;
        response["images"] = images;
        response["task_id"] = task_id;
        response["info"] = info;

        return crow::response(200, response);

    } catch (const NativeRequestConflict& e) {
        return request_error(409, e.what());
    } catch (const std::invalid_argument& e) {
        return request_error(400, e.what());
    } catch (const std::exception& e) {
        const std::string message = sd_generation_error_message(std::string("Generation failed: ") + e.what());
        LOG_ERROR("%s", message.c_str());
        crow::json::wvalue error;
        error["message"] = message;
        task_state_manager_->completeTask(task_id, {}, error.dump());
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleProgress(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }

        if (!json_body.has("id_task")) {
            crow::json::wvalue error;
            error["message"] = "Missing id_task parameter";
            return crow::response(400, error);
        }

        std::string task_id = json_body["id_task"].s();

        if (!task_state_manager_->taskExists(task_id)) {
            crow::json::wvalue error;
            error["message"] = "Task not found";
            return crow::response(404, error);
        }

        bool wants_live_preview = true;
        if (json_body.has("live_preview") && json_body["live_preview"].t() == crow::json::type::True) {
            wants_live_preview = true;
        } else if (json_body.has("live_preview") && json_body["live_preview"].t() == crow::json::type::False) {
            wants_live_preview = false;
        }
        int client_preview_id = -1;
        if (json_body.has("id_live_preview") &&
            json_body["id_live_preview"].t() == crow::json::type::Number) {
            client_preview_id = static_cast<int>(json_body["id_live_preview"].i());
        }
        TaskState state = task_state_manager_->getTaskProgressState(
            task_id, wants_live_preview, client_preview_id);
        const bool has_new_preview = wants_live_preview &&
                                     client_preview_id != state.id_live_preview &&
                                     !state.live_preview.empty();

        crow::json::wvalue response;
        response["completed"] = state.completed;
        response["interrupted"] = state.interrupted;
        response["progress"] = state.progress;
        response["current_step"] = state.current_step;
        response["total_steps"] = state.total_steps;
        response["live_preview"] = has_new_preview ? state.live_preview : "";
        response["id_live_preview"] = state.id_live_preview;

        return crow::response(200, response);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to get progress: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleInterrupt(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body || json_body.t() != crow::json::type::Object ||
            !json_body.has("id_task") || json_body["id_task"].t() != crow::json::type::String) {
            return request_error(400, "A string id_task is required");
        }
        const std::string task_id = json_body["id_task"].s();
        // Hold the identity lock through forwarding so completion cannot admit a
        // new task between checking the ID and selecting the native context.
        std::lock_guard<std::mutex> state_lock(active_task_mutex_);
        if (task_id.empty() || active_task_id_.empty() || task_id != active_task_id_) {
            return request_error(409, "id_task does not identify the active generation");
        }
        const auto state = task_state_manager_->getTaskProgressState(task_id, false, -1);
        if (state.completed || (active_task_kind_ != "text" && state.current_step <= 0)) {
            return request_error(409, "Generation is preparing or already finished; cancellation is unavailable");
        }
        if (active_task_kind_ != "text") image_generator_->interrupt();
        task_state_manager_->interruptTask(task_id);
        LOG_INFO("Matched image or video generation interrupt forwarded");
        return crow::response(200, "OK");
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to interrupt: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleExtraBatchImages(const crow::request& req) {
    if (cosmo_native_main_required())
        return request_error(501, "Native Vulkan HTTP upscaling is not routed through the main-thread executor");
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }

        if (!json_body.has("imageList")) {
            crow::json::wvalue error;
            error["message"] = "Missing imageList parameter";
            return crow::response(400, error);
        }

        // Check if upscaling is requested
        int upscaling_resize = json_body.has("upscaling_resize") ? json_body["upscaling_resize"].i() : 0;

        // Get upscaler name from request (defaults to empty string)
        std::string upscaler_name;
        if (json_body.has("upscaler_1")) {
            std::string webui_upscaler_name = json_body["upscaler_1"].s();
            upscaler_name = convert_webui_upscaler_name(webui_upscaler_name);
        }

        auto image_list = json_body["imageList"];
        std::vector<std::string> result_images;

        if (upscaling_resize > 0) {
            // Collect images into vector
            std::vector<std::string> input_images;
            for (size_t i = 0; i < image_list.size(); i++) {
                input_images.push_back(image_list[i]["data"].s());
            }

            LOG_INFO("Upscaling %zu images with upscaling factor %d using upscaler: %s", input_images.size(),
                     upscaling_resize, upscaler_name.c_str());

            // Use ImageFilters to upscale with specified upscaler
            result_images = image_filters_->upscaleBatch(input_images, upscaler_name, upscaling_resize);
            if (result_images.empty()) {
                crow::json::wvalue error;
                error["message"] = "Upscaler not available. Please configure an upscaler model in options.";
                return crow::response(500, error);
            }
        } else {
            // No upscaling requested, just return the images as-is
            for (size_t i = 0; i < image_list.size(); i++) {
                result_images.push_back(image_list[i]["data"].s());
            }
        }

        crow::json::wvalue response;
        response["images"] = result_images;

        return crow::response(200, response);
    } catch (const std::exception& e) {
        LOG_ERROR("Extra batch images error: %s", e.what());

        crow::json::wvalue error;
        error["message"] = std::string("Failed to process images: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleControlNetDetect(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }

        std::vector<std::string> result_images;

        if (json_body.has("controlnet_input_images")) {
            auto input_images = json_body["controlnet_input_images"];
            std::vector<std::string> base64_images;
            for (size_t i = 0; i < input_images.size(); i++) {
                base64_images.push_back(std::string(input_images[i].s()));
            }

            std::string module = "canny";
            if (json_body.has("controlnet_module")) {
                module = json_body["controlnet_module"].s();
            }

            // Use ImageFilters to apply ControlNet preprocessing
            result_images = image_filters_->applyControlNetFilterBatch(base64_images, module);
        }

        crow::json::wvalue response;
        response["images"] = result_images;

        return crow::response(200, response);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to detect: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleRefreshCheckpoints() {
    try {
        LOG_INFO("Refreshing checkpoints...");
        if (model_manager_) {
            model_manager_->refreshCheckpoints();
        }
        return crow::response(200, "OK");
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to refresh checkpoints: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::handleRefreshVaeAndTextEncoders() {
    try {
        LOG_INFO("Refreshing VAE and text encoders...");
        if (model_manager_) {
            model_manager_->refreshVaeAndTextEncoders();
        }
        return crow::response(200, "OK");
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to refresh VAE and text encoders: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response ApplicationServices::updateSettings(const std::string &body, bool include_options) {
    if (busy()) return request_error(409, "A native generation request is active");
    std::unique_lock<std::mutex> gate(generation_mutex_, std::try_to_lock);
    if (!gate.owns_lock()) return request_error(409, "A native generation request is active");
    try {
        const auto parsed = crow::json::load(body);
        if (!parsed || parsed.t() != crow::json::type::Object)
            throw std::invalid_argument("Settings must be a JSON object");
        if (include_options && parsed.has("options")) {
            if (parsed["options"].t() != crow::json::type::Object)
                throw std::invalid_argument("options must be an object");
            if (parsed["options"].has("sd_model_checkpoint"))
                validateCheckpoint(parsed["options"]["sd_model_checkpoint"], true);
        }
        // Validation, disk replacement and live settings update happen once.
        const auto text = include_options ? cosmo_config_settings_update(body)
                                          : cosmo_config_update(body);
        crow::response result(200, text);
        result.set_header("Content-Type", "application/json");
        result.set_header("Cache-Control", "no-store");
        return result;
    } catch (const std::invalid_argument &error) {
        return request_error(400, error.what());
    } catch (const std::exception &error) {
        return request_error(500, error.what());
    }
}

crow::response ApplicationServices::executeGeneration(int kind, const std::string &body) {
    try {
        crow::request request;
        request.body = kind <= 1 ? cosmo_config_image_request(body) : body;
        switch (kind) {
            case 0: return handleTxt2Img(request);
            case 1: return handleImg2Img(request);
            case 2: return handleTxt2Video(request);
            case 3: return handleImg2Video(request);
            case 4: return generateText(body);
            default: return request_error(400, "Unknown application inference operation");
        }
    } catch (const std::invalid_argument &error) {
        return request_error(400, error.what());
    } catch (const std::exception &error) {
        return request_error(500, sd_generation_error_message(error.what()));
    } catch (...) {
        return request_error(500, "Unknown native generation failure");
    }
}

bool ApplicationServices::submitGeneration(int kind, std::string body,
                                           std::function<void(crow::response)> complete) {
    auto result = std::make_shared<crow::response>();
    return inference_dispatcher_.try_submit(
        [this, kind, body = std::move(body), result] {
            *result = executeGeneration(kind, body);
        },
        [result, complete = std::move(complete)]() mutable {
            complete(std::move(*result));
        });
}

crow::response ApplicationServices::request(cosmo_app_operation operation, const std::string &body) {
    try {
        crow::request request;
        request.body = body;
        switch (operation) {
            case COSMO_APP_STATUS: {
                crow::json::wvalue status;
                status["status"] = admission_closed_.load() ? "closed" : "ready";
                status["busy"] = busy();
                {
                    std::lock_guard<std::mutex> lock(active_task_mutex_);
                    status["active_task_id"] = active_task_id_;
                }
                status["application_id"] = instance_id_;
                status["shared_services"] = true;
                return crow::response(200, status);
            }
            case COSMO_APP_CONFIG_GET: {
                crow::response response(200, cosmo_config_document());
                response.set_header("Content-Type", "application/json");
                response.set_header("Cache-Control", "no-store");
                return response;
            }
            case COSMO_APP_CONFIG_SET: return updateSettings(body, false);
            case COSMO_APP_SETTINGS_SET: return updateSettings(body, true);
            case COSMO_APP_OPTIONS_GET: return handleGetOptions();
            case COSMO_APP_OPTIONS_SET: return handlePostOptions(request);
            case COSMO_APP_DEVICES: return handleBackendDevices();
            case COSMO_APP_MODELS: return handleCheckpoints();
            case COSMO_APP_MODELS_REFRESH: {
                if (busy()) return request_error(409, "A native generation request is active");
                std::unique_lock<std::mutex> gate(generation_mutex_, std::try_to_lock);
                if (!gate.owns_lock()) return request_error(409, "A native generation request is active");
                auto refreshed = handleRefreshCheckpoints();
                return refreshed.code == 200 ? handleCheckpoints() : std::move(refreshed);
            }
            case COSMO_APP_PROGRESS: return handleProgress(request);
            case COSMO_APP_CANCEL: return handleInterrupt(request);
            case COSMO_APP_IMAGE:
            case COSMO_APP_TEXT: {
                struct Completion {
                    std::mutex mutex;
                    std::condition_variable ready;
                    bool done = false;
                    crow::response response;
                };
                auto state = std::make_shared<Completion>();
                if (!submitGeneration(operation == COSMO_APP_TEXT ? 4 : 0, body, [state](crow::response response) {
                    {
                        std::lock_guard<std::mutex> lock(state->mutex);
                        state->response = std::move(response);
                        state->done = true;
                    }
                    state->ready.notify_one();
                })) return request_error(409, "A native generation request is active");
                std::unique_lock<std::mutex> lock(state->mutex);
                state->ready.wait(lock, [&] { return state->done; });
                return std::move(state->response);
            }
        }
        return request_error(400, "Unknown application operation");
    } catch (const std::invalid_argument &error) {
        return request_error(400, error.what());
    } catch (const std::exception &error) {
        return request_error(500, error.what());
    } catch (...) {
        return request_error(500, "Unknown application failure");
    }
}


// Token limits can end between UTF-8 bytes. Keep exact token IDs and raw
// byte count while making the JSON text field valid Unicode for every client.
static std::string application_utf8(const char *bytes, size_t length) {
    std::string output;
    for (size_t i = 0; i < length;) {
        const auto lead = static_cast<unsigned char>(bytes[i]);
        size_t count = lead < 0x80 ? 1 : lead >= 0xc2 && lead <= 0xdf ? 2 :
                       lead >= 0xe0 && lead <= 0xef ? 3 : lead >= 0xf0 && lead <= 0xf4 ? 4 : 0;
        bool valid = count && i + count <= length;
        for (size_t j = 1; valid && j < count; ++j)
            valid = (static_cast<unsigned char>(bytes[i + j]) & 0xc0) == 0x80;
        if (valid && count >= 3) {
            const auto next = static_cast<unsigned char>(bytes[i + 1]);
            valid = !(lead == 0xe0 && next < 0xa0) && !(lead == 0xed && next >= 0xa0) &&
                    !(lead == 0xf0 && next < 0x90) && !(lead == 0xf4 && next >= 0x90);
        }
        if (valid) { output.append(bytes + i, count); i += count; }
        else { output += "\xef\xbf\xbd"; ++i; }
    }
    return output;
}

crow::response ApplicationServices::generateText(const std::string &body) {
    GenerationRequestGuard guard(*this);
    if (!guard.acquired()) return request_error(409, "A native generation request is active");
    std::string task;
    try {
        const auto json = crow::json::load(body);
        if (!json || json.t() != crow::json::type::Object)
            throw std::invalid_argument("Text request must be an object");
        for (const auto &key : json.keys()) {
            if (key != "model" && key != "prompt" && key != "backend" && key != "device" &&
                key != "max_tokens" && key != "threads" && key != "force_task_id")
                throw std::invalid_argument("Unknown text request field: " + key);
        }
        auto text = [&](const char *key, const char *fallback) {
            if (!json.has(key)) return std::string(fallback);
            if (json[key].t() != crow::json::type::String)
                throw std::invalid_argument(std::string(key) + " must be a string");
            std::string value = json[key].s();
            if (value.find('\0') != std::string::npos)
                throw std::invalid_argument(std::string(key) + " contains NUL");
            return value;
        };
        auto number = [&](const char *key, int fallback, int maximum) {
            if (!json.has(key)) return fallback;
            if (json[key].t() != crow::json::type::Number)
                throw std::invalid_argument(std::string(key) + " must be an integer");
            const auto value = json[key].i();
            if (value < 1 || value > maximum || json[key].d() != static_cast<double>(value))
                throw std::invalid_argument(std::string(key) + " is out of range");
            return static_cast<int>(value);
        };
        const auto model = text("model", "/zip/models/stories260K.gguf");
        const auto prompt = text("prompt", "Once upon a time");
        const auto backend = text("backend", cosmo_config_backend());
        const auto device = text("device", backend == cosmo_config_backend() ? cosmo_config_device() : "auto");
        if (backend != "cpu" && backend != "webgpu")
            throw std::invalid_argument("Text backend must be cpu or webgpu");
        if (backend == "cpu" && device != "auto" && device != "CPU")
            throw std::invalid_argument("CPU text backend requires device auto or CPU");
        const int max_tokens = number("max_tokens", 16, 4096);
        const int threads = number("threads", 4, 256);
        guard.start(json, "text");
        task = guard.taskId();
        struct Callbacks {
            TaskStateManager *tasks;
            std::string task;
        } callbacks{task_state_manager_.get(), task};
        cosmo_llama_request request{};
        request.model_path = model.c_str(); request.prompt = prompt.c_str();
        request.backend = backend.c_str(); request.device = device.c_str();
        request.max_tokens = max_tokens;
        request.threads = threads;
        request.user = &callbacks;
        request.cancelled = [](void *opaque) -> int {
            auto &data = *static_cast<Callbacks *>(opaque);
            return data.tasks->getTaskProgressState(data.task, false, -1).interrupted;
        };
        request.progress = [](void *opaque, int generated, int requested) {
            auto &data = *static_cast<Callbacks *>(opaque);
            data.tasks->updateTaskProgress(data.task,
                requested ? static_cast<float>(generated) / requested : 0.0f,
                "", generated, requested);
        };
        struct Result {
            cosmo_llama_result value{};
            ~Result() { cosmo_llama_result_free(&value); }
        } result;
        int status = COSMO_LLAMA_ERROR;
        cosmo_inference_worker([&] {
            status = cosmo_llama_service_generate(&request, &result.value);
            if (status == COSMO_LLAMA_OK)
                return std::vector<std::string>{std::string(result.value.text ? result.value.text : "", result.value.text_bytes)};
            return std::vector<std::string>{};
        });
        if (status == COSMO_LLAMA_INVALID)
            throw std::invalid_argument(result.value.error[0] ? result.value.error : "Invalid text request");
        if (status != COSMO_LLAMA_OK && status != COSMO_LLAMA_CANCELLED)
            throw std::runtime_error(result.value.error[0] ? result.value.error : "Text generation failed");
        if (result.value.cancelled) task_state_manager_->interruptTask(task);
        crow::json::wvalue response;
        response["task_id"] = task;
        response["text"] = application_utf8(result.value.text ? result.value.text : "", result.value.text_bytes);
        response["raw_text_bytes"] = static_cast<uint64_t>(result.value.text_bytes);
        response["cancelled"] = result.value.cancelled != 0;
        response["backend"] = backend;
        const auto *metadata = backend == "webgpu" ? cosmo_webgpu_device_metadata(device.c_str()) : nullptr;
        response["device"] = metadata ? metadata->selector : "CPU";
        response["provider"] = metadata ? metadata->provider : "builtin";
        response["software"] = metadata ? metadata->software != 0 : true;
        response["prompt_tokens"] = result.value.prompt_tokens;
        response["completion_tokens"] = result.value.token_count;
        std::vector<crow::json::wvalue> ids;
        for (int i = 0; i < result.value.token_count; ++i) ids.emplace_back(result.value.token_ids[i]);
        response["token_ids"] = std::move(ids);
        task_state_manager_->completeTask(task, {}, response.dump());
        return crow::response(200, response);
    } catch (const NativeRequestConflict &error) {
        return request_error(409, error.what());
    } catch (const std::invalid_argument &error) {
        return request_error(400, error.what());
    } catch (const std::exception &error) {
        auto response = request_error(500, error.what());
        if (!task.empty()) task_state_manager_->completeTask(task, {}, response.body);
        return response;
    }
}
