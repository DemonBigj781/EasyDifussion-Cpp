#ifndef EASY_DIFFUSION_COSMO_RUNTIME_H
#define EASY_DIFFUSION_COSMO_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

int cosmo_ggml_selftest(void);
int cosmo_shared_ggml_selftest(void);
int cosmo_llama_selftest(void);
int cosmo_llama_webgpu_selftest(void);
int cosmo_webgpu_selftest(void);
int cosmo_llama_generate(int argc, char **argv);
int cosmo_diffusion_selftest(void);
int cosmo_ui_selftest(void);
int cosmo_ui_render(const char *path);
int cosmo_sdkit_main(int argc, char **argv);
int cosmo_train_main(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif
