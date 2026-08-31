# CI architecture

GitHub workflow files are intentionally thin. Hardware identity and architecture mappings live in `ci/hardware/gpus.json`. `ci/scripts/resolve_hardware.py` deduplicates GPU selections, resolves CUDA SM/oneAPI/ROCm targets, and selects the newest mutually compatible CUDA toolkit version. `ci/scripts/build_backend.py` owns backend configure/build behavior. Update the hardware catalog instead of copying mappings into workflows.

Workflow numbering: `00` unified, `01` all-backends test, `10-17` dedicated backends, `20-21` JetPack. All workflows are manual-only.
