# 01 — Python 3.13 and uv-Managed Environments

## Objective

Maintain the main EasyDifussion-Cpp application on Python 3.13 as a lasting
supported runtime. Python cannot be removed while preserving required features:
several integrations use Python-only APIs without easy C++ alternatives. This
work stabilizes retained Python support; it is not preparation for complete
removal. Follow the [Python runtime policy](../python-runtime-policy.md).
Prefer uv over manual venv/pip setup so application and worker dependencies are
managed consistently and tested environments can be frozen and reproduced.

## Implementation

1. Inventory every Python entry point in `ui/`, `scripts/`, plugins, installers, model tools, startup helpers, tests, and subprocess launchers.
2. Inventory imported packages and mark each as: Python-3.13-ready, upgrade required, isolated worker environment required, or candidate for a validated native replacement. Record required Python-only API consumers explicitly.
3. Prefer uv for creating and syncing the authoritative 3.13 application environment. Declare dependencies in `pyproject.toml`, commit `uv.lock`, and pin the Python version. Give incompatible workers separate uv projects and locks; never silently fall back to a system interpreter.
4. Replace hard-coded interpreter paths with an explicit resolver for the application and each worker environment. Preserve virtual-environment paths rather than resolving their symlinks to base interpreters.
5. Add readiness checks that report each selected runtime's Python version and missing/incompatible packages before the UI starts a dependent feature.
6. Update tests to launch through the same application/worker resolvers used in production.
7. After validating a native replacement, remove only packages that have no remaining consumers in that environment. Retain dependencies used by training, plugins, model tools, or compatibility paths.
8. Use locked installation/execution in development, installers, and CI. Support frozen deployments from validated lockfiles without dependency updates; make version upgrades explicit, reviewed lockfile changes.
9. When freezing a controller into a distributable executable, build it from its locked uv environment and document any separate Python worker runtime. Packaging does not remove Python dependencies.

## Likely code areas

`install.sh`, `start.sh`, `ui/easydiffusion/`, `ui/plugins/server/`, `scripts/`,
application/worker `pyproject.toml` and `uv.lock` files, and any subprocess code
that launches Python or uv. Existing `.venv` directories may remain as
uv-managed environments; this objective changes how they are managed.

## Dependencies / ordering

Maintain Python support alongside native development. Native HF/LoRA conversion
can proceed in parallel, but only reduces dependencies for operations with
validated native coverage. It does not eliminate Python or the package sets
needed by other APIs and retained conversion paths.

## Failure behavior

A missing 3.13 application runtime must produce a clear readiness error. A worker
must use its explicitly configured compatible environment and report when that
environment is missing or incompatible. Do not substitute an arbitrary older
system interpreter.

## Validation

Test clean install, upgrade from an existing venv, model tools, plugins, API
startup, and the application test suite on Python 3.13. Validate worker
environments independently, including their readiness checks and coexistence
with the application environment.
Check that a clean uv sync recreates the locked dependency set, that locked
operations reject stale project/lock metadata, and that frozen deployments do
not update dependencies. Verify any frozen executable with its declared worker
environment.

## Complete when

The application runs on its declared Python 3.13 environment, each supported
worker has a declared compatible runtime, and installers, resolvers, and
readiness checks agree. Required Python APIs remain available. No completion
criterion requires removal of Python or forces incompatible workers into the
application environment.
The environments use the documented uv management path, with committed locks,
declared interpreter versions, and reproducible frozen dependency sets.
