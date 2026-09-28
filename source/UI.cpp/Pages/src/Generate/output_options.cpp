#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node output_options() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Output Settings")}),
        Node::element("label", {{"for", "output_format"}}, {Node::text("Output format"),
            Node::element("select", {{"id", "output_format"}, {"name", "output_format"}}, {
                Node::element("option", {{"value", "jpeg"}}, {Node::text("JPEG")}),
                Node::element("option", {{"value", "png"}}, {Node::text("PNG")}),
                Node::element("option", {{"value", "webp"}}, {Node::text("WebP")}),
            })}),
    });
}
}
