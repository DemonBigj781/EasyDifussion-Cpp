#include "easy_diffusion/ui/gallery.hpp"
namespace easy_diffusion::ui::pages::gallery {
namespace {
Node button(const char* id, const char* label) {
    return Node::element("button", {{"id", id}, {"type", "button"}}, {Node::text(label)});
}
}
Node navigate_controls() {
    return Node::element("div", {{"class", "gallery-navigation"}}, {
        button("gallery-prev-page", "← Previous"),
        Node::element("span", {{"id", "gallery-page-status"}, {"aria-live", "polite"}}, {Node::text("Page 1 of 1")}),
        button("gallery-next-page", "Next →"),
    });
}
}
