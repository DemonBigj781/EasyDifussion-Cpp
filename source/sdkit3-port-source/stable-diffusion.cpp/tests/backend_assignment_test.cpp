#include <iostream>
#include <string>

#include "core/ggml_extend_backend.h"

int main() {
    SDBackendManager manager;
    std::string error;
    if (!manager.init("ip-adapter=cpu,clip_vision=cpu,diffusion=cpu,conditioning=cpu,llm=cpu,controlnet=cpu,vae_encode=cpu,vae_decode=cpu,latent_interposer=cpu,latent_interposer_encode=cpu,latent_interposer_decode=cpu", nullptr, nullptr, &error)) {
        std::cerr << "failed to parse module backend assignments: " << error << std::endl;
        return 1;
    }
    if (!manager.runtime_backend_is_cpu(SDBackendModule::IP_ADAPTER)) {
        std::cerr << "IP-Adapter assignment did not resolve to CPU" << std::endl;
        return 1;
    }
    if (!manager.runtime_backend_is_cpu(SDBackendModule::CLIP_VISION)) {
        std::cerr << "CLIP Vision assignment did not resolve to CPU" << std::endl;
        return 1;
    }
    const SDBackendModule routed_modules[] = {
        SDBackendModule::DIFFUSION,
        SDBackendModule::TE,
        SDBackendModule::LLM,
        SDBackendModule::CONTROL_NET,
        SDBackendModule::VAE_ENCODE,
        SDBackendModule::VAE_DECODE,
        SDBackendModule::LATENT_INTERPOSER,
        SDBackendModule::LATENT_INTERPOSER_ENCODE,
        SDBackendModule::LATENT_INTERPOSER_DECODE,
    };
    for (SDBackendModule module : routed_modules) {
        if (!manager.runtime_backend_is_cpu(module)) {
            std::cerr << sd_backend_module_name(module) << " assignment did not resolve to CPU" << std::endl;
            return 1;
        }
    }
    if (std::string(sd_backend_module_name(SDBackendModule::IP_ADAPTER)) != "ip_adapter") {
        std::cerr << "unexpected IP-Adapter module name" << std::endl;
        return 1;
    }
    if (std::string(sd_backend_module_name(SDBackendModule::VAE_ENCODE)) != "vae_encode" ||
        std::string(sd_backend_module_name(SDBackendModule::LATENT_INTERPOSER_DECODE)) != "latent_interposer_decode") {
        std::cerr << "unexpected stage-specific module name" << std::endl;
        return 1;
    }
    return 0;
}
