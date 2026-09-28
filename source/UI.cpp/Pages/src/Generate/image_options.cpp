#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
namespace {
Node number_field(const char* id, const char* label, const char* value) {
    return Node::element("label", {{"for", id}}, {Node::text(label),
        Node::element("input", {{"id", id}, {"name", id}, {"type", "number"}, {"value", value}}, {})});
}
}
Node image_options() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Image Size")}),
        number_field("width", "Width", "512"), number_field("height", "Height", "512"),
    });
}
}
