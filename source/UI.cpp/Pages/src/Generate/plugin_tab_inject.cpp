#include "easy_diffusion/ui/generate.hpp"
#include <utility>
namespace easy_diffusion::ui::pages::generate {
Node plugin_tab_inject() {
    return Node::element("section", {{"id", "generate-plugin-panels"}, {"class", "generate-plugin-panels"}}, {
        Node::element("h3", {}, {Node::text("Generation Plugins")}),
        Node::element("p", {{"class", "muted"}}, {Node::text("Plugin controls can be added here by the C++ page modules.")}),
        lora(), controlnet(),
    });
}
}
