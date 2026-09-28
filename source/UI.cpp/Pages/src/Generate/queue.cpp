#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node queue() {
    return Node::element("div", {{"class", "generation-queue-controls"}}, {
        Node::element("button", {{"id", "makeImage"}, {"type", "button"}, {"class", "primaryButton"}}, {Node::text("Make Image")}),
        Node::element("button", {{"id", "stopImage"}, {"type", "button"}}, {Node::text("Stop")}),
        Node::element("progress", {{"id", "generation-progress"}, {"max", "1"}, {"value", "0"}}, {}),
        Node::element("div", {{"id", "generation-queue-status"}, {"role", "status"}, {"aria-live", "polite"}}, {}),
    });
}
}
