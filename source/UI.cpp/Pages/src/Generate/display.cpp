#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node display() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Image Modifiers")}),
        Node::element("div", {{"id", "editor-inputs-tags-list"}}, {}),
    });
}
}
