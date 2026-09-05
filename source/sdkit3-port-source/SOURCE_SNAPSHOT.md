# sdkit3 source snapshot

This directory is a de-gitted snapshot of the locally patched native sdkit3
source used by EasyDifussion-Cpp.

- Snapshot date: 2026-09-05
- sdkit upstream: <https://github.com/easydiffusion/sdkit>
- sdkit base revision: `de8ff820431358343338ee4363a5fb998e8240a4`
  (`v3.4.1`)
- stable-diffusion.cpp upstream: <https://github.com/leejet/stable-diffusion.cpp>
- stable-diffusion.cpp base revision:
  `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`
- ggml base revision: `e20c3a14aa70ee84ca58499814206dd08d8026bc`

The snapshot includes the upstream stable-diffusion.cpp and GGML changes through
the revisions above together with the local native backend changes present in
the source checkout at refresh time, including the ControlNet, LLLite,
IP-Adapter scheduling, latent-interposer, Flex Attention, TeaCache, xFormers,
SageAttention, image-tool, and vision extensions.

Nested `.git` metadata, generated CMake build directories, Python caches,
runtime `options.json`, editor settings, and private credential files are
intentionally excluded. The original source checkout and its Git history are
not modified by this snapshot.

The nested CMake projects use the recorded snapshot revisions as build-info
fallbacks when their own `.git` metadata is absent. This prevents them from
misidentifying the parent EasyDifussion-Cpp revision as their upstream commit.
