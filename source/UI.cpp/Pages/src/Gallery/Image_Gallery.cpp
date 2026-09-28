#include "easy_diffusion/ui/gallery.hpp"
namespace easy_diffusion::ui::pages::gallery {
Node image_gallery() {
    return Node::element("section", {{"class", "panel-box cpp-image-gallery"}}, {
        Node::element("h3", {}, {Node::text("Image Gallery")}),
        Node::element("div", {{"class", "gallery-settings-row"}}, {
            Node::element("label", {{"for", "gallery-directory-input"}}, {Node::text("Gallery directory")}),
            Node::element("input", {{"id", "gallery-directory-input"}, {"type", "text"}, {"placeholder", "/path/to/image/directory"}}, {}),
            Node::element("button", {{"id", "gallery-directory-save"}, {"type", "button"}, {"class", "primaryButton"}}, {Node::text("Apply")}),
            Node::element("span", {{"id", "gallery-directory-status"}, {"role", "status"}}, {}),
        }),
        options_controls(),
        navigate_controls(),
        load_controls(),
        lightbox_window(),
    });
}
}
