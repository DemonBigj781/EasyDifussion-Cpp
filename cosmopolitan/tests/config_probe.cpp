// Small, model-free driver for the production configuration module.
// Compile this with src/config.cpp. It intentionally does not initialize GPU APIs.
#include "config.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc, char **argv) {
    const int status = cosmo_config_prepare(&argc, &argv);
    if (status >= 0) return status;
    try {
        if (const char *update = std::getenv("COSMO_CONFIG_TEST_UPDATE")) cosmo_config_update(update);
        if (const char *options = std::getenv("COSMO_CONFIG_TEST_OPTIONS")) cosmo_config_set_options(options);
        if (const char *settings = std::getenv("COSMO_CONFIG_TEST_SETTINGS")) cosmo_config_settings_update(settings);
        std::cout << cosmo_config_document() << '\n';
        for (int i = 0; i < argc; ++i) std::cout << "ARG " << argv[i] << '\n';
        std::cout << "POLICY " << cosmo_config_backend() << ' ' << cosmo_config_provider() << ' ' << cosmo_config_device() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
