#pragma once

#include "easy_diffusion/ui.hpp"

#include <string>
#include <vector>

namespace easy_diffusion::ui::pages {

struct Definition {
    std::string id;
    std::string title;
    std::string path;
    std::string group;
    std::string parent;
};

const std::vector<Definition>& definitions();
Page render(const std::string& path);

}  // namespace easy_diffusion::ui::pages
