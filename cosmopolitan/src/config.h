#ifndef COSMO_APP_CONFIG_H
#define COSMO_APP_CONFIG_H
#ifdef __cplusplus
extern "C" {
#endif
/* -1 continues command dispatch; other values are completed command exit codes.
   Call before initializing any WebGPU registry. Returned argv remains owned
   by this module until process exit. CLI overrides never rewrite saved values. */
int cosmo_config_prepare(int *argc, char ***argv);
const char *cosmo_config_provider(void);
const char *cosmo_config_device(void);
const char *cosmo_config_backend(void);
/* Native sdkit calls after enumerating its static registry. Null is a
   reported selection failure; callers must not silently change backends. */
const char *cosmo_config_sdkit_device(void);
#ifdef __cplusplus
}
#endif
#endif
