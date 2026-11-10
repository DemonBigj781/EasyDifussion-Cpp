/* Model-free frontend fixture. No inference or GPU work is performed. */
#include "image_bridge.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int cosmo_image_generate(int, char **);
const char *cosmo_config_backend(void) { return "cpu"; }
const char *cosmo_config_device(void) { return "auto"; }
void sd_ctx_params_init(sd_ctx_params_t *p) { memset(p, 0, sizeof(*p)); p->n_threads = 1; }
void sd_img_gen_params_init(sd_img_gen_params_t *p) { memset(p, 0, sizeof(*p)); }
static const char *path(const char *p) { return p ? p : ""; }
sd_ctx_t *cosmo_image_create_context(const sd_ctx_params_t *p, char *error, size_t size) {
    printf("FIXTURE_MODEL=%s\nFIXTURE_DIFFUSION=%s\nFIXTURE_CLIP_L=%s\nFIXTURE_CLIP_G=%s\nFIXTURE_T5=%s\nFIXTURE_VAE=%s\n",
           path(p->model_path), path(p->diffusion_model_path), path(p->clip_l_path),
           path(p->clip_g_path), path(p->t5xxl_path), path(p->vae_path));
    snprintf(error, size, "fixture stops before model loading");
    return NULL;
}
bool cosmo_image_generate_pixels(sd_ctx_t *c, const sd_img_gen_params_t *p, sd_image_t **i, int *n, char *e, size_t z) { abort(); }
bool sd_ctx_supports_image_generation(const sd_ctx_t *c) { return false; }
void free_sd_ctx(sd_ctx_t *c) {}
void free_sd_images(sd_image_t *i, int n) {}
enum sample_method_t str_to_sample_method(const char *s) { return SAMPLE_METHOD_COUNT; }
enum scheduler_t str_to_scheduler(const char *s) { return SCHEDULER_COUNT; }
enum sample_method_t sd_get_default_sample_method(const sd_ctx_t *c) { return 0; }
enum scheduler_t sd_get_default_scheduler(const sd_ctx_t *c, enum sample_method_t m) { return 0; }
const char *sd_sample_method_name(enum sample_method_t m) { return "fixture"; }
const char *sd_scheduler_name(enum scheduler_t s) { return "fixture"; }
void sd_set_preview_callback(sd_preview_cb_t c, enum preview_t m, int i, bool d, bool n, void *p) {}
void sd_set_sample_progress_callback(sd_progress_cb_t c, void *p) {}
ggml_backend_reg_t ggml_backend_reg_by_name(const char *n) { return (ggml_backend_reg_t)1; }
size_t ggml_backend_reg_dev_count(ggml_backend_reg_t r) { return 1; }
ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t r, size_t n) { return (ggml_backend_dev_t)1; }
const char *ggml_backend_dev_name(ggml_backend_dev_t d) { return "CPU"; }
int64_t ggml_time_us(void) { return 0; }
void *tdefl_write_image_to_png_file_in_memory(const void *p, int w, int h, int c, size_t *s) { abort(); }
void mz_free(void *p) { free(p); }
int main(int argc, char **argv) { return cosmo_image_generate(argc, argv); }
#ifdef COSMO_WEBGPU_BACKEND
#include "cosmo-webgpu.h"
/* A deliberately unavailable provider proves CPU selection avoids opening it. */
int cosmo_webgpu_initialize(void) { puts("FIXTURE_WEBGPU_INITIALIZED"); return -1; }
uint64_t cosmo_webgpu_graph_count(void) { return 0; }
uint64_t cosmo_webgpu_submission_count(void) { return 0; }
uint64_t cosmo_webgpu_dispatch_count(void) { return 0; }
uint64_t cosmo_webgpu_readback_count(void) { return 0; }
uint64_t cosmo_webgpu_matmul_dispatch_count(void) { return 0; }
int cosmo_webgpu_select_device(const char *s) { return -1; }
const struct cosmo_webgpu_device_info *cosmo_webgpu_device_metadata(const char *s) { return NULL; }
const char *cosmo_webgpu_provider_name(void) { return "fixture"; }
const char *cosmo_webgpu_adapter_name(void) { return "fixture"; }
int cosmo_webgpu_adapter_is_software(void) { return 1; }
unsigned long cosmo_webgpu_native_loader_open_count(void) { return 0; }
ggml_backend_dev_t ggml_backend_dev_by_name(const char *n) { return NULL; }
ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t d) { return NULL; }
#endif
