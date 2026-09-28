#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node render_options() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Render Settings")}),
        Node::element("label", {{"for", "steps"}}, {Node::text("Steps"),
            Node::element("input", {{"id", "steps"}, {"name", "steps"}, {"type", "number"}, {"min", "1"}, {"max", "200"}, {"value", "25"}}, {})}),
        Node::element("label", {{"for", "guidance_scale"}}, {Node::text("Guidance scale"),
            Node::element("input", {{"id", "guidance_scale"}, {"name", "guidance_scale"}, {"type", "number"}, {"min", "0"}, {"max", "30"}, {"step", "0.5"}, {"value", "7.5"}}, {})}),
        Node::element("label", {{"for", "sampler"}}, {Node::text("Sampler"),
            Node::element("select", {{"id", "sampler"}, {"name", "sampler"}}, {
                Node::element("option", {{"value", ""}}, {Node::text("Backend default")}),
                Node::element("option", {{"value", "euler_a"}}, {Node::text("Euler a")}),
                Node::element("option", {{"value", "euler"}}, {Node::text("Euler")}),
                Node::element("option", {{"value", "dpm++ 2m"}}, {Node::text("DPM++ 2M")}),
            })}),
    });
}
}
