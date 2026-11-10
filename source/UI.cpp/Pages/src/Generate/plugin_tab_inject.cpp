#include "easy_diffusion/ui/generate.hpp"
#include <utility>
namespace easy_diffusion::ui::pages::generate {
Node plugin_tab_inject() {
    return Node::element("section", {{"id", "generate-plugin-panels"}, {"class", "generate-plugin-panels"}}, {
        Node::element("h3", {}, {Node::text("Generation Plugins")}),
        Node::element("p", {{"class", "muted"}}, {Node::text("Plugin controls can be added here by the C++ page modules.")}),
        Node::element("details", {{"id", "local-llm-prompt-panel"}, {"class", "panel-box generate-plugin-panel"}}, {
            Node::element("summary", {}, {Node::text("Local LLM Prompt Processor")}),
            Node::element("p", {{"id", "local-llm-main-status"}, {"role", "status"}}, {
                Node::text("Checking the configured local LLM…")
            }),
            Node::element("p", {{"class", "muted"}}, {
                Node::text("Process a copy of the current positive prompt with the model and CPU/GPU placement saved in Settings.")
            }),
            Node::element("label", {{"for", "local-llm-instruction"}}, {
                Node::text("Processing instruction"),
                Node::element("textarea", {{"id", "local-llm-instruction"}}, {
                    Node::text("Improve this image-generation prompt while preserving its intent.")
                })
            }),
            Node::element("label", {{"for", "local-llm-source-prompt"}}, {
                Node::text("Prompt to process"),
                Node::element("textarea", {{"id", "local-llm-source-prompt"}, {"placeholder", "Load the current prompt or enter text here."}}, {})
            }),
            Node::element("div", {{"class", "local-llm-actions"}}, {
                Node::element("button", {{"id", "local-llm-load-prompt"}, {"type", "button"}}, {Node::text("Load Current Prompt")}),
                Node::element("button", {{"id", "local-llm-process"}, {"type", "button"}, {"class", "primaryButton"}}, {Node::text("Process Prompt")}),
            }),
            Node::element("label", {{"for", "local-llm-output"}}, {
                Node::text("Processed prompt"),
                Node::element("textarea", {{"id", "local-llm-output"}, {"readonly", "readonly"}, {"placeholder", "The processed prompt will appear here."}}, {})
            }),
            Node::element("button", {{"id", "local-llm-apply"}, {"type", "button"}, {"disabled", "disabled"}}, {
                Node::text("Replace Main Prompt")
            }),
        }),
        lora(), controlnet(),
    });
}
}
