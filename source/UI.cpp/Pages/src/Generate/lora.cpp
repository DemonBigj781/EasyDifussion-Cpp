#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node lora() {
    return Node::element("section", {{"id", "lora-settings-panel"}, {"class", "panel-box generate-plugin-panel"}}, {
        Node::element("h3", {}, {Node::text("LoRA Settings")}),
        Node::element("div", {{"id", "lora_model"}, {"data-path", ""}, {"aria-live", "polite"}}, {
            Node::text("Loading LoRA models…")
        }),
        Node::element("small", {}, {Node::text("Select one or more LoRAs and set their individual strengths.")}),
    });
}
}
