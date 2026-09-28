#pragma once

#include "easy_diffusion/ui.hpp"

namespace easy_diffusion::ui::pages::generate {
Node prompts();
Node display();
Node lora();
Node controlnet();
Node image_options();
Node options();
Node render_options();
Node output_options();
Node queue();
Node image_view();
Node plugin_tab_inject();
Node main_page();
}
