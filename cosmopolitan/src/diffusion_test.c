#include "runtime.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "stable-diffusion.h"

#include <stdio.h>
#include <string.h>

int cosmo_backward_test(void);
int cosmo_optimizer_test(void);

_Static_assert(GGML_MAX_NAME == 160, "All consumers must use the same tensor name size");
_Static_assert((int)GGML_TYPE_F8_E4M3 == (int)SD_TYPE_F8_E4M3, "Diffusion F8 type mismatch");
_Static_assert((int)GGML_TYPE_F8_E5M2 == (int)SD_TYPE_F8_E5M2, "Diffusion F8 type mismatch");

int cosmo_diffusion_selftest(void) {
    const size_t count = sd_get_backend_device_count();
    if (count != ggml_backend_dev_count() || count != 1) {
        fputs("Shared backend registry mismatch\n", stderr);
        return 1;
    }
    sd_backend_device_info_t info;
    if (!sd_get_backend_device_info(0, &info) || info.type != SD_BACKEND_DEVICE_TYPE_CPU ||
        strcmp(info.name, ggml_backend_dev_name(ggml_backend_dev_get(0)))) {
        fputs("Diffusion CPU backend mismatch\n", stderr);
        return 1;
    }
    sd_ctx_params_t model;
    sd_img_gen_params_t image;
    sd_ctx_params_init(&model);
    sd_img_gen_params_init(&image);
    printf("DIFFUSION_API version=%s backend=%s tensor_name_size=%d PASS\n",
           sd_version(), info.name, GGML_MAX_NAME);
    if (cosmo_backward_test() || cosmo_optimizer_test()) return 1;
    puts("TRAINING_MATH_SELFTEST PASS");
    puts("DIFFUSION_SELFTEST PASS");
    return 0;
}
