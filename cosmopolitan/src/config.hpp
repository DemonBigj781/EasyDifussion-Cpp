#ifndef COSMO_APP_CONFIG_HPP
#define COSMO_APP_CONFIG_HPP
#include "config.h"
#include <string>
/* JSON boundary keeps the large parser out of server/model translation units. */
std::string cosmo_config_document();
std::string cosmo_config_update(const std::string &patch);
std::string cosmo_config_options();
void cosmo_config_set_options(const std::string &patch);
#endif
