#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
namespace {
Node size_slider(const char* id, const char* label, const char* value) {
    return Node::element("label", {{"for", id}}, {Node::text(label),
        Node::element("output", {{"id", std::string(id) + "-value"}, {"for", id}}, {Node::text("512 px")}),
        Node::element("input", {{"id", id}, {"name", id}, {"type", "range"}, {"min", "64"},
            {"max", "2048"}, {"step", "64"}, {"value", value}}, {})});
}
}
Node image_options() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Image Size")}),
        size_slider("width", "Width ", "512"), size_slider("height", "Height ", "512"),
    });
}
}
