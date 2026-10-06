#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node prompts() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Prompt")}),
        Node::element("label", {{"for", "prompt"}}, {Node::text("Enter Prompt"),
            Node::element("textarea", {{"id", "prompt"}, {"name", "prompt"}, {"spellcheck", "false"}, {"placeholder", "Describe the image"}}, {})}),
        Node::element("label", {{"for", "negative_prompt"}}, {Node::text("Negative Prompt"),
            Node::element("textarea", {{"id", "negative_prompt"}, {"name", "negative_prompt"}, {"spellcheck", "false"}, {"placeholder", "Optional"}}, {})}),
    });
}
}
