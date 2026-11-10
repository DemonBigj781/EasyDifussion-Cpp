#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node render_options() {
    return Node::element("section", {{"class", "panel-box"}}, {
        Node::element("h3", {}, {Node::text("Render Settings")}),
        Node::element("label", {{"for", "tome_ratio"}}, {Node::text("Spatial token merging (SD1.5 / SDXL)"),
            Node::element("input", {{"id", "tome_ratio"}, {"type", "number"}, {"min", "0"}, {"max", "0.5"}, {"step", "0.05"}, {"value", "0"}}, {})}),
        Node::element("label", {{"for", "negative_guidance_min_sigma"}}, {Node::text("Negative guidance minimum sigma"),
            Node::element("input", {{"id", "negative_guidance_min_sigma"}, {"type", "number"}, {"min", "0"}, {"step", "0.1"}, {"value", "0"}}, {})}),
        Node::element("p", {}, {Node::text("Both default to 0 (off) and can change the image. Token merging reduces spatial self-attention tokens, not frames. Minimum sigma requires standard CFG and disabled video caching. LLLite is incompatible with token merging.")}),
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
