#include "runtime.h"
#ifdef COSMO_WEBGPU_BACKEND
#include "cosmo-webgpu.h"
#endif

#include <stdio.h>
#include <string.h>

static void usage(const char *program) {
    printf("Easy Diffusion - Cosmopolitan Test\n\n"
           "Usage: %s COMMAND [OPTIONS]\n"
           "  --self-test       Check shared tensor, model, training and UI code\n"
           "  ggml-test         Check CPU tensor and quantized operations\n"
           "  webgpu-test       Check embedded WebGPU tensor and model execution\n"
           "  llama [OPTIONS]   Run text generation through the C model API\n"
           "  sdkit [OPTIONS]   Run the native diffusion server or conversion tools\n"
           "  train [OPTIONS]   Run the native SD 1.5 LoRA trainer\n"
           "  render-ui PATH    Render an existing Easy Diffusion UI page\n"
           "  --version         Show this integration build\n",
           program);
}

int main(int argc, char **argv) {
#ifdef COSMO_WEBGPU_BACKEND
    if (cosmo_webgpu_initialize()) {
        fputs("Embedded WebGPU provider initialization failed\n", stderr);
        return 1;
    }
#endif
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        usage(argv[0]);
        return 0;
    }
    if (!strcmp(argv[1], "--version")) {
        puts("Easy Diffusion Cosmopolitan Test; x86-64; static shared GGML");
        return 0;
    }
    if (!strcmp(argv[1], "--self-test")) {
        if (cosmo_ggml_selftest() || cosmo_shared_ggml_selftest() || cosmo_llama_selftest() ||
#ifdef COSMO_WEBGPU_BACKEND
            cosmo_webgpu_selftest() || cosmo_llama_webgpu_selftest() ||
#endif
            cosmo_diffusion_selftest() || cosmo_ui_selftest()) {
            fputs("COSMOPOLITAN_SELFTEST FAIL\n", stderr);
            return 1;
        }
        puts("COSMOPOLITAN_SELFTEST PASS");
        return 0;
    }
    if (!strcmp(argv[1], "ggml-test")) return cosmo_ggml_selftest();
#ifdef COSMO_WEBGPU_BACKEND
    if (!strcmp(argv[1], "webgpu-test")) return cosmo_webgpu_selftest() || cosmo_llama_webgpu_selftest();
#endif
    if (!strcmp(argv[1], "llama")) return cosmo_llama_generate(argc - 1, argv + 1);
    if (!strcmp(argv[1], "sdkit")) return cosmo_sdkit_main(argc - 1, argv + 1);
    if (!strcmp(argv[1], "train")) return cosmo_train_main(argc - 1, argv + 1);
    if (!strcmp(argv[1], "render-ui") && argc == 3) return cosmo_ui_render(argv[2]);
    fprintf(stderr, "Unknown or incomplete command: %s\n", argv[1]);
    usage(argv[0]);
    return 2;
}
