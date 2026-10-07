/* SPDX-License-Identifier: MIT */
#include "cosmo-webgpu.h"
#include "vulkan_loader.h"
#include <atomic>

static std::atomic<uint64_t> graphs{0}, submissions{0}, dispatches{0}, readbacks{0}, matmuls{0};
extern "C" uint64_t cosmo_webgpu_graph_count(void) { return graphs.load(); }
extern "C" uint64_t cosmo_webgpu_submission_count(void) { return submissions.load(); }
extern "C" uint64_t cosmo_webgpu_dispatch_count(void) { return dispatches.load(); }
extern "C" uint64_t cosmo_webgpu_readback_count(void) { return readbacks.load(); }
extern "C" void cosmo_webgpu_note_graph(void) { ++graphs; }
extern "C" void cosmo_webgpu_note_submission(void) { ++submissions; }
extern "C" void cosmo_webgpu_note_dispatch(void) { ++dispatches; }
extern "C" void cosmo_webgpu_note_readback(void) { ++readbacks; }

extern "C" uint64_t cosmo_webgpu_matmul_dispatch_count(void) { return matmuls.load(); }
extern "C" void cosmo_webgpu_note_matmul_dispatch(void) { ++matmuls; }
extern "C" unsigned long cosmo_webgpu_native_loader_open_count(void) { return cosmo_wgpu_vulkan_native_open_count(); }
