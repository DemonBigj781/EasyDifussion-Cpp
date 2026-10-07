/* SPDX-License-Identifier: MIT */
#ifndef COSMO_GGML_WEBGPU_H
#define COSMO_GGML_WEBGPU_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Configure before the first instance; this does not open a driver. */
int cosmo_webgpu_configure(const char *provider, const char *device);
int cosmo_webgpu_initialize(void);
const char *cosmo_webgpu_requested_provider(void);
const char *cosmo_webgpu_requested_device(void);
const char *cosmo_webgpu_last_error(void);
struct cosmo_webgpu_device_info {
    const char *selector;
    const char *provider;
    const char *name;
    const char *stable_id;
    int software;
    uint32_t adapter_type;
    int memory_known;
    uint64_t memory_free;
    uint64_t memory_total;
};
/* Metadata remains alive until process exit. Registry names are not persistent
 * physical identities. Unknown memory has memory_known=0, not maxBufferSize. */
void cosmo_webgpu_register_device_metadata(const char *, const char *, const char *,
                                           int, uint32_t, const char *);
const struct cosmo_webgpu_device_info *cosmo_webgpu_device_metadata(const char *);
size_t cosmo_webgpu_device_count(void);
const struct cosmo_webgpu_device_info *cosmo_webgpu_device_at(size_t);
int cosmo_webgpu_select_device(const char *selector);
const char *cosmo_webgpu_selected_device(void);
/* Resolve saved policy without changing or following the last request. */
const char *cosmo_webgpu_default_device(void);
struct cosmo_webgpu_unavailable_info {
    const char *provider;
    const char *name;
    uint32_t adapter_type;
    const char *reason;
};
void cosmo_webgpu_note_unavailable_adapter(const char *, const char *, uint32_t, const char *);
size_t cosmo_webgpu_unavailable_count(void);
const struct cosmo_webgpu_unavailable_info *cosmo_webgpu_unavailable_at(size_t);
/* Internal factory; descriptor/result are WGPUInstanceDescriptor/WGPUInstance.
 * Serializes the foundation's provider selection with instance creation. */
void *cosmo_webgpu_create_instance(const char *provider, const void *descriptor);
const char *cosmo_webgpu_adapter_name(void);
int cosmo_webgpu_adapter_is_software(void);
uint64_t cosmo_webgpu_graph_count(void);
uint64_t cosmo_webgpu_submission_count(void);
uint64_t cosmo_webgpu_dispatch_count(void);
uint64_t cosmo_webgpu_readback_count(void);
uint64_t cosmo_webgpu_matmul_dispatch_count(void);
const char *cosmo_webgpu_provider_name(void);
unsigned long cosmo_webgpu_native_loader_open_count(void);
/* Instrumentation at actual backend/API call sites, not scheduler estimates. */
void cosmo_webgpu_note_graph(void);
void cosmo_webgpu_note_submission(void);
void cosmo_webgpu_note_dispatch(void);
void cosmo_webgpu_note_readback(void);
void cosmo_webgpu_note_matmul_dispatch(void);
void cosmo_webgpu_note_adapter(const char *name, int software);
#ifdef __cplusplus
}
#endif
#endif
