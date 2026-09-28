#include "easy_diffusion/ui/gallery.hpp"
namespace easy_diffusion::ui::pages::gallery {
namespace {
Node button(const char* id, const char* label, const char* style = "tertiaryButton") {
    return Node::element("button", {{"id", id}, {"type", "button"}, {"class", style}}, {Node::text(label)});
}
}
Node options_controls() {
    return Node::element("div", {{"class", "gallery-options"}}, {
        Node::element("span", {}, {Node::text("Total "), Node::element("span", {{"id", "gallery-count"}}, {Node::text("0")}), Node::text(" images")}),
        button("gallery-select-all", "Select All"), button("gallery-deselect-all", "Deselect All"),
        button("gallery-collage-horizontal", "Horizontal Collage", "primaryButton"),
        button("gallery-collage-vertical", "Vertical Collage", "primaryButton"),
        button("gallery-collage-grid", "Grid Collage", "primaryButton"),
        button("gallery-delete-selected", "Delete Selected"),
        Node::element("label", {{"for", "gallery-zoom-slider"}}, {Node::text("Zoom")}),
        Node::element("input", {{"id", "gallery-zoom-slider"}, {"type", "range"}, {"min", "25"}, {"max", "200"}, {"value", "100"}, {"step", "1"}}, {}),
        Node::element("span", {{"id", "gallery-selection-count"}, {"aria-live", "polite"}}, {}),
    });
}
}
