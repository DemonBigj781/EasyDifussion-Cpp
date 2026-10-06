#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node image_view() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Generated Images")}),
        Node::element("p", {{"id", "initial-text"}}, {Node::text("Enter a prompt, select a checkpoint and choose Generate to begin.")}),
        Node::element("button", {{"id", "clear-all-previews"}, {"type", "button"}}, {Node::text("Clear All")}),
        Node::element("button", {{"id", "show-download-popup"}, {"type", "button"}}, {Node::text("Download images")}),
        Node::element("div", {{"id", "preview-content"}, {"class", "image-grid"}, {"aria-live", "polite"}}, {}),
    });
}
}
