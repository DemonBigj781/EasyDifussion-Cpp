# sdkit3 source snapshot

This directory is a de-gitted snapshot of the locally patched native sdkit3
source used by EasyDifussion-Cpp.

- Snapshot date: 2026-08-29
- sdkit upstream: <https://github.com/easydiffusion/sdkit>
- sdkit base revision: `de8ff820431358343338ee4363a5fb998e8240a4`
  (`v3.4.1`)
- stable-diffusion.cpp upstream: <https://github.com/cmdr2/stable-diffusion.cpp>
- stable-diffusion.cpp base revision:
  `a5ae0fc5134a64092caaa20bb8a0097974e36b1a`
- ggml base revision: `3af5f5760e19a96427f5f7a93b79cbdf3d4b265b`

The snapshot includes the local, uncommitted native backend changes present in
the source checkout at import time, including the ControlNet, LLLite,
IP-Adapter, latent-interposer, Flex Attention, TeaCache, image-tool, and vision
extensions.

Nested `.git` metadata, generated CMake build directories, Python caches,
runtime `options.json`, editor settings, and private credential files are
intentionally excluded. The original source checkout and its Git history are
not modified by this snapshot.

The nested CMake projects use the recorded snapshot revisions as build-info
fallbacks when their own `.git` metadata is absent. This prevents them from
misidentifying the parent EasyDifussion-Cpp revision as their upstream commit.
