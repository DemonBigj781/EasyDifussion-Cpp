#include "easy_diffusion/ui/generate.hpp"
namespace easy_diffusion::ui::pages::generate {
Node controlnet() {
    return Node::element("section", {{"id", "sdkit3-controlnet-panel"}, {"class", "panel-box generate-plugin-panel"}}, {
        Node::element("h3", {}, {Node::text("ControlNet")}),
        Node::element("label", {{"for", "controlnet_mode"}}, {Node::text("Mode"),
            Node::element("select", {{"id", "controlnet_mode"}}, {
                Node::element("option", {{"value", "off"}}, {Node::text("Off")}),
                Node::element("option", {{"value", "standard"}}, {Node::text("Standard ControlNet")}),
                Node::element("option", {{"value", "auto"}}, {Node::text("Automatic Uni / Union")}),
                Node::element("option", {{"value", "lllite"}}, {Node::text("ControlNet-LLLite")}),
            })}),
        Node::element("input", {{"id", "controlnet_enabled"}, {"name", "controlnet_enabled"}, {"type", "checkbox"}, {"hidden", "hidden"}}, {}),
        Node::element("div", {{"id", "controlnet-mode-standard"}, {"data-controlnet-mode", "standard"}}, {
            Node::element("label", {{"for", "controlnet_model"}}, {Node::text("Model"),
                Node::element("input", {{"id", "controlnet_model"}, {"name", "controlnet_model"}, {"type", "text"}, {"list", "controlnet-model-options"}, {"autocomplete", "off"}}, {})}),
            Node::element("datalist", {{"id", "controlnet-model-options"}}, {}),
            Node::element("label", {{"for", "controlnet_alpha"}}, {Node::text("Strength"),
                Node::element("input", {{"id", "controlnet_alpha_slider"}, {"type", "range"}, {"min", "0"}, {"max", "10"}, {"value", "10"}}, {})}),
            Node::element("input", {{"id", "controlnet_alpha"}, {"name", "controlnet_alpha"}, {"type", "number"}, {"min", "0"}, {"max", "10"}, {"step", "0.1"}, {"value", "1"}}, {}),
            Node::element("label", {{"for", "controlnet_union_type"}}, {Node::text("Condition type"),
                Node::element("select", {{"id", "controlnet_union_type"}, {"name", "controlnet_union_type"}}, {
                    Node::element("option", {{"value", "canny"}}, {Node::text("Canny / line art")}),
                    Node::element("option", {{"value", "mlsd"}}, {Node::text("MLSD")}),
                    Node::element("option", {{"value", "softedge"}}, {Node::text("HED / soft edge")}),
                    Node::element("option", {{"value", "sketch"}}, {Node::text("Sketch / scribble")}),
                    Node::element("option", {{"value", "openpose"}}, {Node::text("OpenPose")}),
                    Node::element("option", {{"value", "depth"}}, {Node::text("Depth")}),
                    Node::element("option", {{"value", "normal"}}, {Node::text("Normal")}),
                    Node::element("option", {{"value", "segment"}}, {Node::text("Segmentation")}),
                    Node::element("option", {{"value", "content"}}, {Node::text("Content / global")}),
                })}),
        }),
        Node::element("input", {{"id", "control_image_file"}, {"type", "file"}, {"accept", "image/*"}}, {}),
        Node::element("img", {{"id", "control_image_preview"}, {"alt", "Control image preview"}, {"hidden", "hidden"}}, {}),
        Node::element("input", {{"id", "control_image_filter"}, {"type", "hidden"}, {"value", ""}}, {}),
        Node::element("div", {{"id", "controlnet-extension-modes"}}, {}),
        Node::element("div", {{"id", "controlnet-shared-slot"}}, {}),
    });
}
}
