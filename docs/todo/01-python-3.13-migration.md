# 01 — Python 3.13 Migration

## Objective
Move every Python-dependent EasyDifussion-Cpp workflow onto Python 3.13 before removing Python from core paths. This is a stabilization phase, not the final architecture.

## Implementation
1. Inventory every Python entry point in `ui/`, `scripts/`, plugins, installers, model tools, startup helpers, tests, and subprocess launchers.
2. Inventory imported packages and mark each as: Python-3.13-ready, upgrade required, replacement required, or scheduled for native removal.
3. Make the installer create one authoritative 3.13 environment and stop silently falling back to another system interpreter.
4. Replace hard-coded interpreter paths with one runtime resolver. Preserve the virtual-environment path rather than resolving its symlink to the base interpreter.
5. Add a readiness command that reports Python version plus missing/incompatible packages before the UI starts a dependent feature.
6. Update tests to launch through the same interpreter resolver used in production.
7. Once a Python subsystem gains a native replacement, remove its dependency from the 3.13 environment rather than leaving dead packages installed.

## Likely code areas
`install.sh`, `start.sh`, `ui/easydiffusion/`, `ui/plugins/server/`, `scripts/`, and any subprocess code that launches `python` or `python3`.

## Dependencies / ordering
Do this before complete Python removal. Native HF/LoRA conversion can proceed in parallel and should eventually eliminate its own Python package set.

## Failure behavior
A missing 3.13 runtime must produce a clear readiness error. Do not silently run an older Python version because that creates environment-dependent bugs.

## Validation
Test clean install, upgrade from an existing venv, model tools, plugins, API startup, and test suite on Python 3.13.

## Complete when
No supported Python workflow requires an older interpreter, the installer/readiness checks agree on one runtime, and unsupported packages are either replaced or explicitly documented.
