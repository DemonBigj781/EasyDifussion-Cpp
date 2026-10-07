/* SPDX-License-Identifier: MIT */
#ifndef COSMO_APPLICATION_H
#define COSMO_APPLICATION_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cosmo_app cosmo_app;
typedef enum cosmo_app_operation {
    COSMO_APP_STATUS,
    COSMO_APP_CONFIG_GET,
    COSMO_APP_CONFIG_SET,
    COSMO_APP_OPTIONS_GET,
    COSMO_APP_OPTIONS_SET,
    COSMO_APP_SETTINGS_SET,
    COSMO_APP_DEVICES,
    COSMO_APP_MODELS,
    COSMO_APP_MODELS_REFRESH,
    COSMO_APP_IMAGE,
    COSMO_APP_PROGRESS,
    COSMO_APP_CANCEL,
    COSMO_APP_TEXT
} cosmo_app_operation;

/* Configuration must already be prepared. Create/destroy on the original
   main thread, outside run(). The application owns one shared model index,
   inference engine, task table and request admission gate. */
cosmo_app *cosmo_app_create(char *error, size_t error_size);
void cosmo_app_destroy(cosmo_app *app);

/* Returns a HTTP-like status (200, 400, 409, 500); reply is an owned JSON
   string, including errors. No exception crosses this C boundary.
   IMAGE waits for its actual shared inference job. Run it from the client
   callback while run() pumps the native main-thread lane. Other callers may
   concurrently query progress or cancel the matching task ID. */
int cosmo_app_request(cosmo_app *app, cosmo_app_operation operation,
                     const char *request_json, char **reply_json);
void cosmo_app_reply_free(char *reply_json);

typedef int (*cosmo_app_client)(cosmo_app *app, void *user);
/* Runs a client, optionally alongside the HTTP transport sharing this same
   application. A NULL client serves HTTP until stop; client return drains
   outstanding work and stops HTTP before returning its result. Each app has
   one run lifecycle; create/run/destroy it once on the original main thread. */
int cosmo_app_run(cosmo_app *app, int serve_http, cosmo_app_client client,
                  void *user, char *error, size_t error_size);
void cosmo_app_stop(cosmo_app *app);

#ifdef __cplusplus
}
#endif
#endif
