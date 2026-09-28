#include "easy_diffusion/ui/generate.hpp"
#include <utility>
#include <vector>
namespace easy_diffusion::ui::pages::generate {
namespace {
Node two_columns(Node left, Node right) {
    return Node::element("div", {{"class", "generate-columns"}}, {
        Node::element("section", {{"class", "generate-controls-column"}}, {std::move(left)}),
        Node::element("section", {{"class", "generate-preview-column"}}, {std::move(right)}),
    });
}
}
Node main_page() {
    Node controls = Node::element("div", {{"class", "generation-controls"}}, {
        prompts(), display(), plugin_tab_inject(), image_options(), options(),
        render_options(), output_options(), queue(),
    });
    return two_columns(std::move(controls), image_view());
}
}
