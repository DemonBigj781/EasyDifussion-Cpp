#include "server.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "image_utils.h"
#include "logging.h"

// Custom log handler for Crow that filters out /ping requests
class FilteredLogHandler : public crow::ILogHandler {
   public:
    void log(std::string message, crow::LogLevel level) override {
        // Skip logging /ping requests to reduce noise
        if (message.find("/ping") != std::string::npos || message.find("/internal/progress") != std::string::npos) {
            return;
        }

        // Use our existing logging system for consistency
        LogLevel our_level;
        switch (level) {
            case crow::LogLevel::Debug:
                our_level = LogLevel::Debug;
                break;
            case crow::LogLevel::Info:
                our_level = LogLevel::Info;
                break;
            case crow::LogLevel::Warning:
                our_level = LogLevel::Warning;
                break;
            case crow::LogLevel::Error:
            case crow::LogLevel::Critical:
                our_level = LogLevel::Error;
                break;
            default:
                our_level = LogLevel::Info;
                break;
        }

        log_message(our_level, "[CROW] %s", message.c_str());
    }
};

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
        {"IPNDM", "ipndm"},
        {"IPNDM_V", "ipndm_v"},
        {"LCM", "lcm"},
        {"DDIM", "ddim_trailing"},
        {"TCD", "tcd"},
    };

    auto it = mapping.find(name);
    if (it != mapping.end()) return it->second;
    return name;
}

static std::string convert_webui_scheduler_name(const std::string& name) {
    static const std::unordered_map<std::string, std::string> mapping = {
        {"automatic", "discrete"},      {"uniform", "discrete"},           {"karras", "karras"},
        {"exponential", "exponential"}, {"sgm_uniform", "sgm_uniform"},    {"simple", "simple"},
        {"align_your_steps", "ays"},    {"align_your_steps_GITS", "gits"},
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

Server::Server(const ServerParams& params)
    : params_(params), port_(params.port), model_manager_(params.model_manager), should_stop_(false) {
    // Set up custom logger to filter out unnecessary requests
    static FilteredLogHandler filtered_handler;
    crow::logger::setHandler(&filtered_handler);

    options_manager_ = std::make_shared<OptionsManager>();
    task_state_manager_ = std::make_shared<TaskStateManager>();

    // Create ImageFilters for upscaling and other image processing
    image_filters_ = std::make_shared<ImageFilters>(model_manager_);

    // Create ImageGenerator with shared task state manager and model manager
    image_generator_ =
        std::make_unique<ImageGenerator>(task_state_manager_, options_manager_, model_manager_, image_filters_, params);

    options_manager_->load();

    setupRoutes();
}

Server::~Server() { stop(); }

void Server::setupRoutes() {
    // Ping endpoint
    CROW_ROUTE(app_, "/v1/internal/ping").methods("GET"_method)([this]() { return handlePing(); });

    // Options endpoints
    CROW_ROUTE(app_, "/v1/sdapi/v1/options").methods("GET"_method)([this]() { return handleGetOptions(); });

    CROW_ROUTE(app_, "/v1/sdapi/v1/options").methods("POST"_method)([this](const crow::request& req) {
        return handlePostOptions(req);
    });

    // Image generation endpoints
    CROW_ROUTE(app_, "/v1/sdapi/v1/txt2img").methods("POST"_method)([this](const crow::request& req) {
        return handleTxt2Img(req);
    });

    CROW_ROUTE(app_, "/v1/sdapi/v1/img2img").methods("POST"_method)([this](const crow::request& req) {
        return handleImg2Img(req);
    });

    CROW_ROUTE(app_, "/v1/sdapi/v1/txt2video").methods("POST"_method)([this](const crow::request& req) {
        return handleTxt2Video(req);
    });

    CROW_ROUTE(app_, "/v1/sdapi/v1/img2video").methods("POST"_method)([this](const crow::request& req) {
        return handleImg2Video(req);
    });

    // Progress endpoint
    CROW_ROUTE(app_, "/v1/internal/progress").methods("POST"_method)([this](const crow::request& req) {
        return handleProgress(req);
    });

    // Interrupt endpoint
    CROW_ROUTE(app_, "/v1/sdapi/v1/interrupt").methods("POST"_method)([this](const crow::request& req) {
        return handleInterrupt(req);
    });

    // Extra batch images endpoint
    CROW_ROUTE(app_, "/v1/sdapi/v1/extra-batch-images").methods("POST"_method)([this](const crow::request& req) {
        return handleExtraBatchImages(req);
    });

    // ControlNet detect endpoint
    CROW_ROUTE(app_, "/v1/controlnet/detect").methods("POST"_method)([this](const crow::request& req) {
        return handleControlNetDetect(req);
    });

    // Refresh endpoints
    CROW_ROUTE(app_, "/v1/sdapi/v1/refresh-checkpoints").methods("POST"_method)([this]() {
        return handleRefreshCheckpoints();
    });

    CROW_ROUTE(app_, "/v1/sdapi/v1/refresh-vae-and-text-encoders").methods("POST"_method)([this]() {
        return handleRefreshVaeAndTextEncoders();
    });
}

void Server::run() {
    std::cout << "Starting server on port " << port_ << std::endl;
    app_.bindaddr("127.0.0.1").port(port_).multithreaded().run();
}

void Server::stop() {
    should_stop_ = true;
    app_.stop();
}

crow::response Server::handlePing() { return crow::response(200, "OK"); }

crow::response Server::handleGetOptions() {
    try {
        auto options = options_manager_->getOptions();
        return crow::response(200, options);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to get options: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handlePostOptions(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["message"] = "Invalid JSON";
            return crow::response(400, error);
        }

        if (options_manager_->setOptions(json_body)) {
            return crow::response(200, "OK");
        } else {
            crow::json::wvalue error;
            error["message"] = "Failed to save options";
            return crow::response(500, error);
        }
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to set options: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleTxt2Img(const crow::request& req) {
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

crow::response Server::handleImg2Img(const crow::request& req) {
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

crow::response Server::handleTxt2Video(const crow::request& req) {
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

crow::response Server::handleImg2Video(const crow::request& req) {
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

crow::response Server::generateVideo(const crow::json::rvalue& json_body, bool is_img2video) {
    std::string task_id = json_body.has("force_task_id")
                              ? std::string(json_body["force_task_id"].s())
                              : "default_video_task";
    task_state_manager_->createTask(task_id);

    try {
        VideoGenerationParams params;
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

        std::vector<std::string> frames = image_generator_->generateVideo(params, task_id);
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
        response["fps"]    = params.fps;
        response["info"]   = std::move(info);
        return crow::response(200, response);
    } catch (const std::exception& e) {
        LOG_ERROR("Video generation error: %s", e.what());
        crow::json::wvalue error;
        error["message"] = std::string("Video generation failed: ") + e.what();
        task_state_manager_->completeTask(task_id, {}, error.dump());
        return crow::response(500, error);
    }
}

crow::response Server::generateImage(const crow::json::rvalue& json_body, bool is_img2img) {
    // Extract task_id
    std::string task_id = "default_task";
    if (json_body.has("force_task_id")) {
        task_id = json_body["force_task_id"].s();
    }

    // Create task
    task_state_manager_->createTask(task_id);

    try {
        // Parse generation parameters
        ImageGenerationParams params;
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

        // Generate images (runs in same thread, blocks until complete)
        std::vector<std::string> images;
        if (is_img2img) {
            images = image_generator_->generateImg2Img(params, task_id);
        } else {
            images = image_generator_->generateTxt2Img(params, task_id);
        }

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
        response["info"] = info;

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR("Image generation error: %s", e.what());
        crow::json::wvalue error;
        error["message"] = std::string("Generation failed: ") + e.what();
        task_state_manager_->completeTask(task_id, {}, error.dump());
        return crow::response(500, error);
    }
}

crow::response Server::handleProgress(const crow::request& req) {
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

        TaskState state = task_state_manager_->getTaskState(task_id);

        crow::json::wvalue response;
        response["completed"] = state.completed;
        response["progress"] = state.progress;
        response["current_step"] = state.current_step;
        response["total_steps"] = state.total_steps;
        response["live_preview"] = state.live_preview;
        response["id_live_preview"] = state.id_live_preview;

        return crow::response(200, response);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to get progress: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleInterrupt(const crow::request& req) {
    try {
        // Interrupt the image generator
        if (image_generator_) {
            image_generator_->interrupt();
            LOG_INFO("Image or video generation interrupt forwarded");
        }

        return crow::response(200, "OK");
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["message"] = std::string("Failed to interrupt: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleExtraBatchImages(const crow::request& req) {
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

crow::response Server::handleControlNetDetect(const crow::request& req) {
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

crow::response Server::handleRefreshCheckpoints() {
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

crow::response Server::handleRefreshVaeAndTextEncoders() {
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
