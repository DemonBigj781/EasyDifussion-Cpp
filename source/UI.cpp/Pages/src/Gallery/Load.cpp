#include "easy_diffusion/ui/gallery.hpp"
namespace easy_diffusion::ui::pages::gallery {
namespace {
Node button(const char* id, const char* label) {
    return Node::element("button", {{"id", id}, {"type", "button"}}, {Node::text(label)});
}
}
Node load_controls() {
    return Node::element("div", {{"class", "gallery-load-controls"}}, {
        button("gallery-refresh", "Refresh Directory"),
        Node::element("p", {{"id", "gallery-source"}, {"role", "status"}}, {Node::text("Loading configured gallery directory…")}),
        Node::element("div", {{"id", "gallery-container"}, {"class", "cpp-gallery-grid"}, {"aria-live", "polite"}}, {}),
        Node::element("p", {{"id", "gallery-empty"}, {"hidden", "hidden"}}, {Node::text("No supported images were found in the configured gallery directory.")}),
    });
}
}
