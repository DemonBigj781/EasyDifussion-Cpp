#include "easy_diffusion/ui/gallery.hpp"
namespace easy_diffusion::ui::pages::gallery {
Node lightbox_window() {
    return Node::element("dialog", {{"id", "gallery-lightbox"}, {"class", "cpp-gallery-lightbox"}}, {
        Node::element("button", {{"id", "gallery-lightbox-close"}, {"type", "button"}, {"aria-label", "Close preview"}}, {Node::text("×")}),
        Node::element("button", {{"id", "gallery-lightbox-prev"}, {"type", "button"}, {"aria-label", "Previous image"}}, {Node::text("←")}),
        Node::element("img", {{"id", "gallery-lightbox-image"}, {"alt", "Full-size gallery preview"}}, {}),
        Node::element("button", {{"id", "gallery-lightbox-next"}, {"type", "button"}, {"aria-label", "Next image"}}, {Node::text("→")}),
        Node::element("p", {{"id", "gallery-lightbox-counter"}}, {}),
    });
}
}
