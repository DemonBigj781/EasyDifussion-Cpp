# llama.cpp source snapshot

This directory is a de-gitted snapshot of the official llama.cpp repository
used by EasyDifussion-Cpp's Model Tools GGUF converter.

- Snapshot date: 2026-08-29
- Upstream: <https://github.com/ggml-org/llama.cpp>
- Revision: `c589f0ed10c643678c4707dd160c21ac7633ebc0`
- Upstream commit: `metal : add fa-vec tunings for M2 (#27940)`
- License: MIT; see `LICENSE` in this directory

The upstream `.git` directory and generated build/cache directories are not
included. No tracked submodules were present at the recorded revision. The
remaining source tree, conversion scripts, tests, documentation, third-party
notices, and vocabulary fixtures are preserved as supplied upstream.

The CMake build-info helpers and embedded ggml build use the recorded revision
when `.git` metadata is absent. This prevents a de-gitted build from reporting
the parent EasyDifussion-Cpp revision as either its llama.cpp or ggml revision.
