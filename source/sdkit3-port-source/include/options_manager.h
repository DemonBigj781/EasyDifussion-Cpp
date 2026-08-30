#ifndef __OPTIONS_MANAGER_H__
#define __OPTIONS_MANAGER_H__

#include <mutex>
#include <string>

#include "crow.h"

class OptionsManager {
   public:
    OptionsManager(const std::string& options_file = "options.json");
    ~OptionsManager();

    // Load options from file
    bool load();

    // Save options to file
    bool save();

    // Get all options as JSON
    crow::json::wvalue getOptions();

    // Set options from JSON
    bool setOptions(const crow::json::rvalue& options);

   private:
    std::string options_file_;
    crow::json::wvalue options_data_;
    std::mutex mutex_;
};

#endif  // __OPTIONS_MANAGER_H__
