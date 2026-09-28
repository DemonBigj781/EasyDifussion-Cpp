#include "easy_diffusion/ui/settings.hpp"

#include <string>
#include <utility>
#include <vector>

namespace easy_diffusion::ui::pages::settings {
namespace {
using Module = std::pair<const char*, const char*>;
Node device_group(const char* scope, const char* title, const std::vector<Module>& modules) {
    std::vector<Node> controls;
    for (const auto& module : modules) {
        const std::string id = std::string("cpp-") + scope + "-device-" + module.first;
        controls.push_back(Node::element("label", {{"for", id}}, {
            Node::text(module.second),
            Node::element("select", {{"id", id}, {"data-native-scope", scope}, {"data-native-module", module.first}}, {
                Node::element("option", {{"value", ""}}, {Node::text("Automatic")})
            })
        }));
    }
    return Node::element("fieldset", {{"class", "native-device-routing-group"}}, {
        Node::element("legend", {}, {Node::text(title)}),
        Node::element("div", {{"class", "native-device-routing-grid"}}, std::move(controls))
    });
}
}
Node native_device_routing() {
    return Node::element("div", {{"class", "native-device-routing-settings"}}, {
        Node::element("p", {{"id", "native-device-routing-status"}, {"role", "status"}}, {
            Node::text("Loading native compute devices…")
        }),
        device_group("image", "Image pipeline", {
            {"diffusion", "KSampler / denoising"}, {"te", "Conditioning / text encoders"},
            {"llm", "LLM conditioning"}, {"vae_encode", "VAE encoding"}, {"vae_decode", "VAE decoding"},
            {"controlnet", "ControlNet / LLLite"}, {"clip_vision", "CLIP-Vision encoding"},
            {"ip_adapter", "IP-Adapter projection"}, {"photomaker", "Identity encoding"},
            {"upscaler", "Latent / model upscaling"}, {"detector", "Detection / detailing"},
            {"latent_interposer_encode", "Encode latent interposer"},
            {"latent_interposer_decode", "Decode latent interposer"}
        }),
        device_group("video", "Video pipeline", {
            {"diffusion", "KSampler / video denoising"}, {"te", "Conditioning / text encoders"},
            {"llm", "LLM conditioning"}, {"vae_encode", "Video VAE encoding"},
            {"vae_decode", "Video VAE decoding"}, {"clip_vision", "Image / CLIP-Vision encoding"},
            {"latent_interposer_encode", "Encode latent interposer"},
            {"latent_interposer_decode", "Decode latent interposer"}
        }),
        Node::element("button", {{"id", "native-device-routing-refresh"}, {"type", "button"}}, {Node::text("Refresh devices")})
    });
}
}
