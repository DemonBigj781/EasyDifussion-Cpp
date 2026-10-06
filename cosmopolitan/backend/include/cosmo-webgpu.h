/* SPDX-License-Identifier: MIT */
#ifndef COSMO_GGML_WEBGPU_H
#define COSMO_GGML_WEBGPU_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Call before GGML registry enumeration. Failure never selects a native driver. */
int cosmo_webgpu_initialize(void);
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
