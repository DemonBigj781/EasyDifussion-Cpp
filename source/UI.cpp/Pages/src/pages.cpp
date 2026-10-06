#include "easy_diffusion/ui/pages.hpp"
#include "easy_diffusion/ui/settings.hpp"
#include "easy_diffusion/ui/generate.hpp"
#include "easy_diffusion/ui/gallery.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace easy_diffusion::ui::pages {
namespace {

const std::vector<Definition> kDefinitions = {
    {"main", "Main", "/", "primary", ""},
    {"canvas", "Canvas", "/canvas", "primary", ""},
    {"perchance-text", "Text", "/perchance/text", "perchance", "perchance"},
    {"perchance-image", "Image", "/perchance/image", "perchance", "perchance"},
    {"perchance-gallery", "Gallery", "/perchance/gallery", "perchance", "perchance"},
    {"image-gallery", "Image Gallery", "/gallery/images", "galleries", "galleries"},
    {"dataset-gallery", "Dataset Gallery", "/gallery/datasets", "galleries", "galleries"},
    {"tagging", "Tagging", "/tagging", "primary", ""},
    {"training", "Training", "/training", "primary", ""},
    {"model-downloading", "Model Downloading", "/models/download", "primary", ""},
    {"settings", "Settings", "/settings", "system", ""},
    {"gpu-config", "GPU Config", "/settings/gpu", "system", ""},
    {"plugin-config", "Plugin Config", "/settings/plugins", "system", ""},
    {"stats", "Stats", "/stats", "system", ""},
    {"logs", "Logs", "/logs", "system", ""},
    {"console", "Console", "/console", "system", ""},
};

const Definition* find_path(const std::string& path) {
    const auto found = std::find_if(kDefinitions.begin(), kDefinitions.end(), [&path](const Definition& item) {
        return item.path == path;
    });
    return found == kDefinitions.end() ? nullptr : &*found;
}

Node link(const Definition& definition, const Definition* active) {
    const std::string href = definition.path == "/" ? "/cpp-ui" : "/cpp-ui" + definition.path;
    std::vector<Attribute> attributes = {{"href", href}};
    if (active && definition.id == active->id) attributes.push_back({"aria-current", "page"});
    return Node::element("a", std::move(attributes), {Node::text(definition.title)});
}

Node nav(const char* label, const std::string& group, const Definition* active) {
    std::vector<Node> links;
    for (const auto& definition : kDefinitions) {
        if (definition.group == group && definition.parent.empty())
            links.push_back(Node::element("li", {}, {link(definition, active)}));
    }
    if (group == "primary") {
        for (const auto& section : {std::pair<const char*, const char*>{"Perchance", "perchance"},
                                    {"Galleries", "galleries"}}) {
            std::vector<Node> children;
            const Definition* first = nullptr;
            for (const auto& definition : kDefinitions) {
                if (definition.parent != section.second) continue;
                if (!first) first = &definition;
                children.push_back(Node::element("li", {}, {link(definition, active)}));
            }
            if (first) {
                links.push_back(Node::element("li", {}, {
                    Node::element("a", {{"href", "/cpp-ui" + first->path}}, {Node::text(section.first)}),
                    Node::element("ul", {}, std::move(children)),
                }));
            }
        }
    }
    return Node::element("nav", {{"aria-label", label}},
        {Node::element("ul", {}, std::move(links))});
}

Node children_nav(const std::string& parent, const Definition* active) {
    std::vector<Node> links;
    for (const auto& definition : kDefinitions) {
        if (definition.parent == parent)
            links.push_back(Node::element("li", {}, {link(definition, active)}));
    }
    return Node::element("nav", {{"aria-label", "Page section"}},
        {Node::element("ul", {}, std::move(links))});
}

Node button(const char* id, const char* label, const char* style = "tertiaryButton") {
    return Node::element("button", {{"id", id}, {"type", "button"}, {"class", style}}, {Node::text(label)});
}

Node input(const char* id, const char* label, const char* type = "text", const char* value = "") {
    std::vector<Attribute> attrs = {{"id", id}, {"name", id}, {"type", type}};
    if (*value) attrs.push_back({"value", value});
    return Node::element("label", {{"for", id}}, {
        Node::text(label), Node::element("input", std::move(attrs))
    });
}

Node textarea(const char* id, const char* label, const char* placeholder = "") {
    return Node::element("label", {{"for", id}}, {
        Node::text(label), Node::element("textarea", {{"id", id}, {"name", id}, {"placeholder", placeholder}}, {})
    });
}

Node panel(const char* title, std::vector<Node> children) {
    children.insert(children.begin(), Node::element("h3", {}, {Node::text(title)}));
    return Node::element("section", {{"class", "panel-box"}}, std::move(children));
}

Node select_input(const char* id, const char* label, const std::vector<std::pair<std::string, std::string>>& options) {
    std::vector<Node> choices;
    for (const auto& option : options)
        choices.push_back(Node::element("option", {{"value", option.first}}, {Node::text(option.second)}));
    return Node::element("label", {{"for", id}}, {
        Node::text(label), Node::element("select", {{"id", id}, {"name", id}}, std::move(choices))
    });
}

Node image_grid(const char* id) {
    return Node::element("div", {{"id", id}, {"class", "image-grid"}, {"aria-live", "polite"}}, {});
}

Node page_content(const Definition& page) {
    if (page.id == "main") return generate::main_page();
    if (page.id == "canvas") {
        return panel("Canvas", {
            Node::element("div", {{"class", "canvas-toolbar"}}, {
                button("canvas-draw", "Draw"), button("canvas-inpaint", "Inpaint"), button("canvas-generate", "Generate")
            }),
            Node::element("canvas", {{"id", "image-editor-canvas"}, {"aria-label", "Canvas editor"}}, {}),
            Node::element("div", {{"id", "image-editor-page-content"}}, {}),
        });
    }
    if (page.id == "perchance-text") {
        return panel("Perchance Text", {
            Node::element("p", {{"data-perchance-status", ""}, {"role", "status"}}, {Node::text("Checking launcher…")}),
            textarea("perchance-generator-text-prompt", "Text prompt", "Describe the text to generate"),
            select_input("perchance-generator-text-preset", "Prompt starter preset", {{"", "Choose a preset…"}}),
            button("perchance-generator-apply-text-preset", "Insert into Start With"),
            input("perchance-generator-text-preset-name", "Custom starter name"),
            button("perchance-generator-save-text-preset", "Save Starter"),
            button("perchance-generator-import-text-prompt", "Import Prompt"),
            button("perchance-generator-text-button", "Generate Text", "primaryButton"),
            textarea("perchance-generator-start-with", "Start with"),
            textarea("perchance-generator-stops", "Stop sequences"),
            textarea("perchance-generator-filters", "Filter out strings"),
            Node::element("textarea", {{"id", "perchance-generator-text-output"}, {"readonly", "readonly"}}, {}),
        });
    }
    if (page.id == "perchance-image") {
        return panel("Perchance Image", {
            textarea("perchance-generator-image-prompt", "Positive prompt", "Describe the image"),
            textarea("perchance-generator-negative-prompt", "Negative prompt"),
            select_input("perchance-generator-shape", "Shape", {{"square", "Square"}, {"portrait", "Portrait"}, {"landscape", "Landscape"}}),
            input("perchance-generator-seed", "Seed", "number", "-1"),
            input("perchance-generator-guidance-scale", "Guidance scale", "number", "7"),
            input("perchance-generator-amount", "Generation amount", "number", "1"),
            button("perchance-generator-image-button", "Generate Image", "primaryButton"),
            image_grid("perchance-generator-image-grid"),
        });
    }
    if (page.id == "perchance-gallery") {
        return panel("Perchance Gallery", {
            Node::element("p", {{"data-perchance-gallery-status", ""}, {"role", "status"}}, {Node::text("Checking content policy…")}),
            input("perchance-generator-gallery-id", "Gallery image ID or URL"),
            input("perchance-generator-gallery-channel", "Generator channel", "text", "ai-text-to-image-generator"),
            input("perchance-generator-gallery-limit", "Limit", "number", "20"),
            select_input("perchance-generator-gallery-sort", "Sort", {{"recent", "Recent"}, {"top", "Top"}, {"trending", "Trending"}}),
            button("perchance-generator-gallery-list-button", "List Gallery", "primaryButton"),
            button("perchance-generator-gallery-get-button", "Get Image", "primaryButton"),
            image_grid("perchance-generator-gallery-results"),
        });
    }
    if (page.id == "dataset-gallery") {
        Node search = Node::element("fieldset", {}, {
            Node::element("legend", {}, {Node::text("Search images")}),
            textarea("training-scrape-query", "Search tags", "e.g. blue_hair"),
            Node::element("datalist", {{"id", "training-scrape-tag-suggestions"}}, {}),
            input("training-scrape-dataset", "Gallery dataset", "text", "grabber"),
            Node::element("label", {{"for", "training-scrape-sources"}}, {Node::text("Websites")}),
            Node::element("select", {{"id", "training-scrape-sources"}, {"multiple", "multiple"}, {"size", "8"}}, {}),
            button("grabber-gallery-sites-all", "Select all"), button("grabber-gallery-sites-none", "Clear selection"),
            input("training-scrape-limit", "Maximum results per site", "number", "80"),
            input("training-scrape-threshold", "Near duplicate cutoff", "number", "0.95"),
            Node::element("p", {{"id", "training-scrape-source-note"}, {"role", "status"}}, {Node::text("Loading available websites automatically…")}),
            button("training-scrape-search", "Search and download", "primaryButton"),
            Node::element("p", {{"id", "grabber-gallery-status"}, {"role", "status"}, {"aria-live", "polite"}}, {Node::text("Loading websites…")}),
        });
        Node search_results = panel("Search results", {
            button("grabber-gallery-select-all", "Select all results"),
            button("grabber-gallery-download", "Download selected"),
            Node::element("progress", {{"id", "grabber-gallery-progress"}}, {}),
            image_grid("grabber-gallery-results"),
        });
        Node attempt = Node::element("fieldset", {}, {
            Node::element("legend", {}, {Node::text("Scrape attempts")}),
            select_input("training-scrape-attempts", "Attempt", {{"", "Select an attempt"}}),
            button("training-scrape-refresh-attempts", "Refresh attempts"),
            button("training-scrape-load-attempt", "Open attempt"),
            Node::element("p", {{"id", "training-scrape-status"}, {"role", "status"}}, {Node::text("Search results and downloads are grouped by attempt.")}),
        });
        Node processed = panel("Processed images", {
            Node::element("div", {{"class", "gallery-source-tabs"}}, {
                button("grabber-gallery-stills-tab", "Still images"),
                button("grabber-gallery-frames-tab", "GIF/video frames"),
            }),
            image_grid("training-scrape-filtered"),
            image_grid("grabber-gallery-frames"),
            Node::element("div", {{"id", "training-scrape-before"}, {"hidden", "hidden"}}, {}),
        });
        Node selected = panel("Selected for dataset", {image_grid("grabber-gallery-selected")});
        Node gallery_columns = Node::element("div", {{"class", "grabber-gallery-columns"}}, {
            Node::element("section", {}, {std::move(processed)}),
            Node::element("section", {}, {std::move(selected)}),
        });
        Node tag_settings = Node::element("fieldset", {}, {
            Node::element("legend", {}, {Node::text("Auto-tag settings")}),
            input("training-tagger", "WD14 model", "text", "wd-v1-4-moat-tagger-v2"),
            input("training-threshold", "Tag threshold", "number", "0.35"),
            input("training-character-threshold", "Character threshold", "number", "0.85"),
            input("training-exclude", "Exclude tags (comma separated)"),
            button("grabber-gallery-tag", "Auto-tag gallery"),
            button("training-scrape-tag", "Re-run tagging"),
        });
        Node trigger = Node::element("fieldset", {{"class", "grabber-trigger-tags"}}, {
            Node::element("legend", {}, {Node::text("LoRA trigger tag")}),
            input("grabber-gallery-trigger-tags", "", "text", ""),
        });
        return Node::element("div", {{"class", "grabber-gallery"}}, {
            search, search_results, attempt, gallery_columns, tag_settings, trigger,
            Node::element("div", {{"class", "gallery-actions"}}, {
                button("training-scrape-delete", "Delete selected"),
                button("training-scrape-archive", "Submit selected to training dataset", "primaryButton"),
            }),
            Node::element("p", {}, {Node::text("Submitted images are processed into a dataset for the Trainer.")}),
            Node::element("label", {{"for", "training-archived-datasets"}}, {Node::text("Archived datasets")}),
            Node::element("select", {{"id", "training-archived-datasets"}}, {}),
        });
    }
    if (page.id == "image-gallery") return gallery::image_gallery();
    if (page.id == "training") {
        auto group = [](const char* title, std::vector<Node> fields) {
            return Node::element("fieldset", {{"class", "training-group"}}, {
                Node::element("legend", {}, {Node::text(title)}),
                Node::element("div", {{"class", "training-fields"}}, std::move(fields)),
            });
        };
        return Node::element("div", {{"class", "training-layout"}}, {
        Node::element("div", {{"class", "training-setup"}}, {
        panel("Training setup", {
            Node::element("p", {{"id", "training-readiness"}, {"role", "status"}}, {Node::text("Checking training runtime…")}),
            button("training-refresh", "Refresh runtime and models"),
            group("1 · Dataset and model", {
            select_input("training-dataset", "Dataset", {{"", "Load a dataset"}}),
            input("training-trigger", "Optional extra tag (blank uses captions only)"),
            select_input("training-kind", "Train", {{"lora", "LoRA"}, {"embedding", "Embedding"}}),
            select_input("training-architecture", "Base model family", {{"sd15", "Stable Diffusion 1.5"}, {"sdxl", "SDXL"}, {"anima", "Anima"}}),
            select_input("training-backend", "Training backend", {{"native", "Native C++ (SD1.5)"}, {"python", "Legacy Python (sd-scripts)"}}),
            input("training-model", "Full checkpoint (search models)"),
            input("training-name", "Output name", "text", "my_lora"),
            }),
            group("2 · Training and learning rates", {
            input("training-steps", "Training steps", "number", "1200"),
            input("training-epochs", "Epochs (optional; overrides steps)", "number"),
            input("training-repeats", "Dataset repeats per epoch", "number", "1"),
            input("training-resolution", "Resolution", "number", "512"),
            input("training-batch", "Batch size", "number", "1"),
            input("training-rate", "Learning rate", "number", "0.00005"),
            input("training-text-encoder-rate", "Text encoder learning rate (SD1.5 LoRA; 0 freezes CLIP)", "number", "0.000005"),
            input("training-rank", "LoRA rank", "number", "16"),
            input("training-alpha", "LoRA alpha (blank follows rank)", "number", ""),
            select_input("training-scheduler", "Learning rate schedule", {{"constant", "Constant"}, {"cosine_with_restarts", "Cosine with hard restarts"}}),
            input("training-warmup", "Warmup steps", "number", "0"),
            input("training-cycles", "Cosine cycles (1 = no intermediate restart)", "number", "1"),
            select_input("training-precision", "Precision", {{"fp16", "FP16"}, {"bf16", "BF16"}, {"no", "FP32"}}),
            input("training-save-every", "Save every N steps", "number", "100"),
            input("training-seed", "Seed", "number", "42"),
            }),
            Node::element("p", {{"id", "training-native-note"}, {"class", "training-note"}}, {
                Node::text("Native SD1.5 uses AdamW, batch size 1 and fixed precision. Each full epoch is images × repeats steps; its checkpoint and fixed-seed caption sample finish before the next epoch starts."),
            }),
            Node::element("div", {{"id", "training-anima-settings"}, {"hidden", "hidden"}}, {
            group("Anima settings", {
            select_input("training-qwen3", "Qwen3 text encoder", {{"", "Select configured text encoder"}}),
            select_input("training-vae", "Qwen-Image VAE", {{"", "Select configured VAE"}}),
            select_input("training-checkpointing", "Gradient checkpointing", {{"standard", "Standard"}, {"cpu_offload", "CPU offload"}, {"unsloth", "Unsloth asynchronous offload"}, {"off", "Disabled"}}),
            input("training-blocks-swap", "Anima blocks to swap", "number", "0"),
            select_input("training-optimizer", "Optimizer", {{"AdamW", "AdamW"}, {"AdamW8bit", "AdamW8bit"}}),
            Node::element("label", {{"for", "training-cache-text"}}, {
                Node::element("input", {{"id", "training-cache-text"}, {"type", "checkbox"}, {"checked", "checked"}}),
                Node::text(" Cache Anima text encoder outputs"),
            }),
            }),
            }),
            Node::element("p", {{"class", "training-note"}}, {Node::text("Finish generation and release loaded inference models before starting training.")}),
            button("training-start", "Start training", "primaryButton"),
            Node::element("p", {{"id", "training-error"}, {"role", "alert"}}, {}),
        }),
        }),
        Node::element("aside", {{"class", "training-results"}, {"aria-label", "Training jobs and runtimes"}}, {
        panel("Jobs and results", {
            select_input("training-job", "Job history", {{"", "Select a job"}}),
            Node::element("p", {{"id", "training-job-status"}, {"role", "status"}}, {Node::text("No job selected.")}),
            Node::element("progress", {{"id", "training-progress"}, {"aria-label", "Training progress"}, {"max", "1"}, {"value", "0"}}, {}),
            Node::element("div", {{"class", "training-actions"}}, {
                button("training-cancel", "Cancel job"), button("training-resume", "Resume LoRA"),
            }),
            Node::element("h4", {}, {Node::text("Job log")}),
            Node::element("pre", {{"id", "training-log"}, {"tabindex", "0"}, {"aria-label", "Training log"}}, {Node::text("Select a job to view its log.")}),
        }),
        Node::element("details", {{"class", "panel-box training-sprite"}}, {
            Node::element("summary", {}, {Node::text("SpriteGPT · separate runtime")}),
            Node::element("p", {}, {Node::text("64×64 RGB sprite training, separate from LoRA training. No VAE is used.")}),
            Node::element("p", {{"id", "training-spritegpt-status"}, {"role", "status"}}, {Node::text("Checking SpriteGPT runtime…")}),
            Node::element("div", {{"class", "training-actions"}}, {
                button("training-spritegpt-start", "Start SpriteGPT runtime"),
                button("training-spritegpt-open", "Open SpriteGPT Studio"),
                button("training-spritegpt-stop", "Stop SpriteGPT runtime"),
            }),
        }),
        }),
        });
    }
    if (page.id == "model-downloading") {
        return panel("Online Model Browser", {
            select_input("civitai-downloader-provider", "Provider", {{"civitai", "Civitai"}, {"huggingface", "Hugging Face"}}),
            input("civitai-downloader-query", "Search models", "search"),
            button("civitai-downloader-search", "Search", "primaryButton"),
            select_input("civitai-downloader-sort", "Sort", {{"", "Default"}}),
            select_input("civitai-downloader-period", "Period", {{"AllTime", "All time"}, {"Month", "Month"}, {"Week", "Week"}}),
            Node::element("div", {{"id", "civitai-downloader-status"}, {"role", "status"}}, {}),
            Node::element("div", {{"id", "civitai-downloader-results"}}, {}),
        });
    }
    if (page.id == "plugin-config") {
        return panel("Local Plugin Manager", {
            Node::element("p", {}, {Node::text("The same optional plugins and saved choices as the legacy UI. Changes are saved automatically in this browser. Reload existing legacy pages to apply changes made here.")}),
            Node::element("p", {}, {Node::text("All listed plugins have modern UI support. Enable a plugin here, then open Main for its controls and image actions. Perchance plugins use their dedicated pages.")}),
            input("cpp-plugin-filter", "Search plugins", "search"),
            Node::element("div", {{"id", "cpp-plugin-manager-controls"}}, {}),
        });
    }
    if (page.id == "settings" || page.id == "gpu-config") {
        std::vector<Node> settings = {
            Node::element("div", {{"id", "system-settings-table"}, {"class", "parameters-table"}}, {
                Node::element("div", {{"data-setting-id", "backend_platform"}, {"data-save-in-app-config", "true"}}, {
                    Node::element("label", {{"for", "backend_platform"}}, {Node::text("Backend platform")} ),
                    Node::element("select", {{"id", "backend_platform"}}, {
                        Node::element("option", {{"value", "auto"}}, {Node::text("Automatic (CUDA / ROCm, otherwise Vulkan)")}),
                        Node::element("option", {{"value", "auto-cuda"}}, {Node::text("Auto CUDA")} ),
                        Node::element("option", {{"value", "auto-vulkan"}}, {Node::text("Auto Vulkan")} ),
                        Node::element("option", {{"value", "cuda"}}, {Node::text("NVIDIA CUDA")} ),
                        Node::element("option", {{"value", "rocm"}}, {Node::text("AMD ROCm")} ),
                        Node::element("option", {{"value", "vulkan"}}, {Node::text("Vulkan")} ),
                        Node::element("option", {{"value", "sycl"}}, {Node::text("Intel oneAPI / SYCL")} ),
                    })
                })
            }),
            Node::element("div", {{"id", "system-settings-extra"}}, {}),
        };
        if (page.id == "gpu-config") {
            settings.push_back(settings::native_device_routing());
        }
        if (page.id == "settings") {
            settings.push_back(Node::element("fieldset", {{"id", "cpp-input-saving"}}, {}));
            settings.push_back(Node::element("fieldset", {{"id", "kiosk-settings"}}, {
                Node::element("legend", {}, {Node::text("Kiosk mode")}),
                Node::element("label", {{"for", "kiosk-mode-enabled"}}, {
                    Node::element("input", {{"id", "kiosk-mode-enabled"}, {"type", "checkbox"}}),
                    Node::text(" Enable kiosk mode"),
                }),
                Node::element("p", {}, {Node::text("Only SD1.5, SDXL and Anima base checkpoints; no LoRAs. Perchance uses its stricter G filter; unrated generation is unavailable. The main gallery uses Destockd, never local images.")}),
                Node::element("p", {}, {Node::text("This is a display policy, not a password-protected lock. Destockd's source filtering is not a PG certification.")}),
                button("kiosk-mode-save", "Save kiosk mode", "primaryButton"),
                Node::element("p", {{"id", "kiosk-mode-save-status"}, {"role", "status"}}, {}),
            }));
        }
        settings.push_back(button("save-system-settings-btn", "Save", "primaryButton"));
        settings.push_back(Node::element("div", {{"id", "system-info"}}, {
            Node::text("System information will appear here.")
        }));
        return panel(page.title.c_str(), std::move(settings));
    }
    if (page.id == "tagging") {
        return panel("Dataset Tagging", {
            select_input("tagging-dataset", "Dataset", {{"", "Select a dataset"}}),
            select_input("tagging-model", "Tagger", {{"wd-v1-4-moat-tagger-v2", "WD14 Moat v2"}}),
            input("tagging-threshold", "Tag threshold", "number", "0.35"),
            button("tagging-run", "Autotag missing captions", "primaryButton"),
            image_grid("tagging-items"),
        });
    }
    if (page.id == "logs") {
        return panel("Application and Backend Log", {
            Node::element("p", {{"id", "log-viewer-status"}, {"role", "status"}}, {Node::text("Loading recent application logs…")}),
            button("log-viewer-refresh", "Refresh logs", "primaryButton"),
            Node::element("pre", {{"id", "log-viewer-output"}, {"aria-live", "polite"}}, {}),
        });
    }
    if (page.id == "console") {
        return panel("Diagnostics Console", {
            Node::element("p", {{"id", "ui-console-status"}, {"role", "status"}}, {Node::text("Read-only app diagnostics and file navigation. Type help for commands. Symlinks are allowed; no shell or editor.")}),
            Node::element("form", {{"id", "ui-console-form"}}, {
                Node::element("label", {{"for", "ui-console-command"}}, {Node::text("Command"),
                    Node::element("input", {{"id", "ui-console-command"}, {"type", "text"}, {"autocomplete", "off"},
                        {"spellcheck", "false"}, {"placeholder", "help · status · logs · ls · cd models · stat config.yaml"}})}),
                Node::element("button", {{"id", "ui-console-run"}, {"type", "submit"}, {"class", "primaryButton"}}, {Node::text("Run")}),
            }),
            button("ui-console-clear", "Clear console"),
            Node::element("pre", {{"id", "ui-console-output"}, {"aria-live", "polite"}}, {}),
        });
    }
    if (page.id == "stats") {
        return panel(page.title.c_str(), {Node::element("pre", {{"id", "system-output"}, {"aria-live", "polite"}}, {})});
    }
    return Node::element("p", {}, {Node::text("This page is not connected yet.")});
}

}  // namespace

const std::vector<Definition>& definitions() {
    return kDefinitions;
}

Page render(const std::string& path) {
    const Definition* active = find_path(path);
    if (!active) throw std::out_of_range("Unknown UI page path: " + path);

    std::vector<Node> shell = {
        Node::element("header", {}, {
            Node::element("h1", {}, {Node::text("Easy Diffusion C++")}),
            Node::element("p", {{"id", "kiosk-status-banner"}, {"role", "status"}, {"hidden", "hidden"}}, {}),
            nav("Primary navigation", "primary", active),
            nav("System navigation", "system", active),
        }),
    };
    if (!active->parent.empty()) shell.push_back(children_nav(active->parent, active));
    shell.push_back(Node::element("main", {{"id", "page-content"}, {"data-page", active->id}}, {
        Node::element("h2", {}, {Node::text(active->title)}),
        page_content(*active),
    }));

    Page page(active->title + " · Easy Diffusion C++");
    page.add_stylesheet("/cpp-ui/assets/ui.css")
        .add_stylesheet("/media/css/themes.css")
        .add_stylesheet("/media/css/fonts.css")
        .set_body(Node::element("div", {{"class", "app-shell"}}, std::move(shell)));
    page.add_script("/media/js/local-plugin-preferences.js");
    if (active->id == "main" || active->id == "settings")
        page.add_script("/cpp-ui/scripts/input_preferences.js");
    page.add_script("/cpp-ui/scripts/kiosk.js");
    page.add_script("/cpp-ui/scripts/plugin_preferences.js");
    if (active->id == "logs") page.add_script("/cpp-ui/assets/log-viewer.js");
    if (active->id == "console") page.add_script("/cpp-ui/assets/console.js");
    if (active->id == "gpu-config") page.add_script("/cpp-ui/scripts/native-device-routing.js");
    if (active->id == "settings" || active->id == "gpu-config")
        page.add_script("/cpp-ui/scripts/backend-platform.js");
    if (active->id == "plugin-config") page.add_script("/cpp-ui/scripts/plugin_manager.js");
    if (active->id == "main" || active->id == "training") {
        page.add_stylesheet("/media/css/searchable-models.css");
        page.add_stylesheet("/media/css/fontawesome-all.min.css");
        page.add_script("/media/js/utils.js");
        page.add_script("/media/js/searchable-models.js");
    }
    if (active->id == "main") {
        page.add_script("/media/js/ip-adapter-compatibility.js");
        page.add_script("/cpp-ui/scripts/ip_adapter.js");
        page.add_script("/cpp-ui/scripts/generate_plugins.js");
        page.add_script("/cpp-ui/scripts/image_modifiers.js");
        page.add_script("/cpp-ui/scripts/generation_queue.js");
        page.add_script("/cpp-ui/scripts/optional_plugins.js");
        page.add_script("/cpp-ui/scripts/generate.js");
        page.add_script("/cpp-ui/scripts/creative_plugins.js");
        page.add_script("/cpp-ui/scripts/workbench_core.js");
        page.add_script("/cpp-ui/scripts/prompt_workbench.js");
        page.add_script("/cpp-ui/scripts/image_workbench.js");
        page.add_script("/cpp-ui/scripts/template_manager.js");
        page.add_script("/cpp-ui/scripts/spell_workbench.js");
        page.add_script("/plugins/core/prompt_plugin/prompt-assist.plugin.js");
    }
    if (active->id == "image-gallery") page.add_script("/cpp-ui/scripts/gallery.js");
    if (active->id == "perchance-gallery") page.add_script("/cpp-ui/scripts/perchance_gallery.js");
    if (active->id == "training") page.add_script("/cpp-ui/scripts/training.js");
    return page;
}

}  // namespace easy_diffusion::ui::pages
