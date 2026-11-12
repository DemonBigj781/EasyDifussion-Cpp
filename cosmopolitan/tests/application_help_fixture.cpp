/* Model-free startup fixture. No backend or application service is linked. */
#include "application.h"
#include "runtime.h"
#include "shell.h"
#include "cosmo-webgpu.h"
#include <cstdio>
#include <cstdlib>

namespace {
int unexpected() { std::fputs("FIXTURE_UNEXPECTED_CALL\n", stderr); std::exit(90); }
}

extern "C" {
int cosmo_webgpu_configure(const char *, const char *) {
    std::puts("FIXTURE_BACKEND_CONFIGURE");
    return 0;
}
int cosmo_webgpu_initialize(void) {
    std::puts("FIXTURE_BACKEND_INITIALIZE");
    return 0;
}
int cosmo_webgpu_select_device(const char *) { return unexpected(); }
const char *cosmo_webgpu_selected_device(void) { unexpected(); return nullptr; }
cosmo_app *cosmo_app_create(char *error, size_t size) {
    std::puts("FIXTURE_APPLICATION_CREATE");
    std::snprintf(error, size, "fixture stops before application services");
    return nullptr;
}
void cosmo_app_destroy(cosmo_app *) { unexpected(); }
void cosmo_app_reply_free(char *) { unexpected(); }
int cosmo_app_request(cosmo_app *, cosmo_app_operation, const char *, char **) { return unexpected(); }
int cosmo_app_run(cosmo_app *, int, cosmo_app_client, void *, char *, size_t) { return unexpected(); }
int cosmo_shell_run(int, char **, const cosmo_shell_services *) { return unexpected(); }
int cosmo_ggml_selftest(void) { return unexpected(); }
int cosmo_shared_ggml_selftest(void) { return unexpected(); }
int cosmo_llama_selftest(void) { return unexpected(); }
int cosmo_llama_webgpu_selftest(void) { return unexpected(); }
int cosmo_webgpu_selftest(void) { return unexpected(); }
int cosmo_webgpu_device_test(int, char **) { return unexpected(); }
int cosmo_webgpu_inplace_selftest(void) { return unexpected(); }
int cosmo_llama_generate(int, char **) { return unexpected(); }
int cosmo_image_generate(int, char **) { return unexpected(); }
int cosmo_diffusion_selftest(void) { return unexpected(); }
int cosmo_ui_selftest(void) { return unexpected(); }
int cosmo_ui_render(const char *) { return unexpected(); }
int cosmo_sdkit_main(int, char **) { return unexpected(); }
int cosmo_train_main(int, char **) { return unexpected(); }
}
