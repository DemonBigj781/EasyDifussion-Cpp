#include "options_manager.h"

#include <fstream>
#include <iostream>

OptionsManager::OptionsManager(const std::string& options_file) : options_file_(options_file) {}

OptionsManager::~OptionsManager() { save(); }

bool OptionsManager::load() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ifstream file(options_file_);
    if (!file.is_open()) {
        // File doesn't exist yet, start with default options
        options_data_["sd_model_checkpoint"] = "";
        options_data_["live_previews_enable"] = true;
        options_data_["CLIP_stop_at_last_layers"] = -1;
        options_data_["sdxl_clip_l_skip"] = false;
        options_data_["samples_format"] = "png";
        options_data_["forge_additional_modules"] = std::vector<std::string>{};
        return true;
    }

    // Read file contents
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (content.empty()) {
        // Empty file, use defaults
        options_data_["sd_model_checkpoint"] = "";
        options_data_["live_previews_enable"] = true;
        options_data_["CLIP_stop_at_last_layers"] = -1;
        options_data_["sdxl_clip_l_skip"] = false;
        options_data_["samples_format"] = "png";
        options_data_["forge_additional_modules"] = std::vector<std::string>{};
        return true;
    }

    // Parse JSON
    try {
        auto json_data = crow::json::load(content);
        if (!json_data) {
            std::cerr << "Failed to parse options file" << std::endl;
            return false;
        }

        // Copy loaded data to options_data_
        for (const auto& key : json_data.keys()) {
            switch (json_data[key].t()) {
                case crow::json::type::String:
                    options_data_[key] = json_data[key].s();
                    break;
                case crow::json::type::Number:
                    options_data_[key] = json_data[key].d();
                    break;
                case crow::json::type::True:
                    options_data_[key] = true;
                    break;
                case crow::json::type::False:
                    options_data_[key] = false;
                    break;
                case crow::json::type::List: {
                    std::vector<std::string> vec;
                    for (size_t i = 0; i < json_data[key].size(); i++) {
                        vec.push_back(json_data[key][i].s());
                    }
                    options_data_[key] = vec;
                    break;
                }
                default:
                    break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception loading options: " << e.what() << std::endl;
        return false;
    }

    return true;
}

bool OptionsManager::save() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(options_file_);
    if (!file.is_open()) {
        std::cerr << "Failed to open options file for writing: " << options_file_ << std::endl;
        return false;
    }

    // Manually build JSON string (simple approach)
    file << options_data_.dump();
    file.close();

    return true;
}

crow::json::wvalue OptionsManager::getOptions() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Return a copy of the options
    crow::json::wvalue result;

    //  Manually copy each key
    std::string json_str = options_data_.dump();
    auto parsed = crow::json::load(json_str);

    for (const auto& key : parsed.keys()) {
        switch (parsed[key].t()) {
            case crow::json::type::String:
                result[key] = parsed[key].s();
                break;
            case crow::json::type::Number:
                result[key] = parsed[key].d();
                break;
            case crow::json::type::True:
                result[key] = true;
                break;
            case crow::json::type::False:
                result[key] = false;
                break;
            case crow::json::type::List: {
                std::vector<std::string> vec;
                for (size_t i = 0; i < parsed[key].size(); i++) {
                    vec.push_back(parsed[key][i].s());
                }
                result[key] = vec;
                break;
            }
            default:
                break;
        }
    }

    return result;
}

bool OptionsManager::setOptions(const crow::json::rvalue& options) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Update options_data_ with new values
    for (const auto& key : options.keys()) {
        switch (options[key].t()) {
            case crow::json::type::String:
                options_data_[key] = options[key].s();
                break;
            case crow::json::type::Number:
                options_data_[key] = options[key].d();
                break;
            case crow::json::type::True:
                options_data_[key] = true;
                break;
            case crow::json::type::False:
                options_data_[key] = false;
                break;
            case crow::json::type::List: {
                std::vector<std::string> vec;
                for (size_t i = 0; i < options[key].size(); i++) {
                    vec.push_back(options[key][i].s());
                }
                options_data_[key] = vec;
                break;
            }
            default:
                break;
        }
    }

    // Save to file
    std::ofstream file(options_file_);
    if (!file.is_open()) {
        return false;
    }

    file << options_data_.dump();
    file.close();

    return true;
}
