#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node display() {
    return Node::element("section", {{"id", "cpp-image-modifiers"}, {"class", "panel-box"}, {"aria-labelledby", "cpp-modifier-heading"}}, {
        Node::element("h3", {{"id", "cpp-modifier-heading"}}, {Node::text("Image Modifiers")}),
        Node::element("p", {{"class", "cpp-modifier-note"}}, {Node::text("Choose styles to include with your prompt. Click a selected style again to remove it.")}),
        Node::element("div", {{"class", "cpp-modifier-filters"}}, {
            Node::element("label", {{"for", "cpp-modifier-search"}}, {
                Node::text("Filter modifiers"),
                Node::element("input", {{"id", "cpp-modifier-search"}, {"type", "search"}, {"placeholder", "Search styles, artists, lighting…"}}),
            }),
            Node::element("label", {{"for", "cpp-modifier-category"}}, {
                Node::text("Category"),
                Node::element("select", {{"id", "cpp-modifier-category"}}, {
                    Node::element("option", {{"value", ""}}, {Node::text("All categories")}),
                }),
            }),
        }),
        Node::element("div", {{"id", "editor-inputs-tags-list"}, {"aria-label", "Selected modifiers"}}, {}),
        Node::element("div", {{"class", "cpp-modifier-actions"}}, {
            Node::element("button", {{"id", "cpp-modifier-clear"}, {"type", "button"}, {"disabled", "disabled"}}, {Node::text("Clear selected")}),
            Node::element("button", {{"id", "cpp-modifier-retry"}, {"type", "button"}, {"hidden", "hidden"}}, {Node::text("Retry loading modifiers")}),
        }),
        Node::element("p", {{"id", "cpp-modifier-status"}, {"role", "status"}, {"aria-live", "polite"}}, {Node::text("Loading image modifiers…")}),
        Node::element("div", {{"id", "cpp-modifier-grid"}, {"class", "cpp-modifier-grid"}, {"aria-busy", "true"}}, {}),
    });
}
}
