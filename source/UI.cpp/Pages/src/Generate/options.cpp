#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node options() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Generation Options")}),
        Node::element("label", {{"for", "stable_diffusion_model"}}, {Node::text("Checkpoint"),
            Node::element("select", {{"id", "stable_diffusion_model"}, {"name", "stable_diffusion_model"}}, {
                Node::element("option", {{"value", ""}}, {Node::text("Loading checkpoints…")})
            })}),
        Node::element("p", {{"id", "generation-model-status"}, {"role", "status"}}, {}),
        Node::element("label", {{"for", "num_images"}}, {Node::text("Images"),
            Node::element("input", {{"id", "num_images"}, {"name", "num_images"}, {"type", "number"}, {"min", "1"}, {"max", "8"}, {"value", "1"}}, {})}),
        Node::element("label", {{"for", "seed"}}, {Node::text("Seed (-1 for random)"),
            Node::element("input", {{"id", "seed"}, {"name", "seed"}, {"type", "number"}, {"value", "-1"}}, {})}),
    });
}
}
