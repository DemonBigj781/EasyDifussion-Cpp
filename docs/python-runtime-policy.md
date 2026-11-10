# Python is a retained project dependency

## Decision — 2026-09-20

Python cannot be removed from Easy Diffusion while preserving the project's
required feature and API coverage. Complete Python removal is not a plausible
project objective and must not be described as an inevitable final stage of
native development.

Several required integrations expose their supported functionality through
Python APIs and workflows, with no easy, maintained, feature-equivalent C++
alternatives. Replacing them would require substantial reimplementation and
ongoing compatibility work. A native inference backend does not replace these
application, training, and model-tool dependencies.

This policy supersedes earlier roadmap language promising complete Python
removal. It is an architecture decision, not a claim that the experimental
Theory branch is ready to run.

## Preferred environment management: uv

Prefer uv over manually assembled venv/pip workflows for consistency across
development, installers, CI, and packaged deployments. Retaining Python means
maintaining reproducible environments, not relying on whichever packages happen
to be installed on a machine.

Declare dependencies in `pyproject.toml`, commit the generated `uv.lock`, and
pin the Python interpreter version for each application or worker environment.
Record the intended platform and accelerator/package variant alongside those
pins. Training and other APIs with incompatible requirements should have
separate uv projects and lockfiles, managed through the same conventions.

Environment freezing is an explicit requirement: a tested dependency set must
be reproducible without silently resolving newer versions. Normal installation
and execution should use locked operations such as `uv sync --locked` and
`uv run --locked`, which reject an outdated lockfile instead of updating it.
For deployments deliberately consuming an already validated lockfile,
`--frozen` can use that lockfile without updating it or checking whether it
matches changed project metadata. Dependency upgrades should be intentional
lockfile changes followed by validation.

uv can still create a `.venv` directory; that directory is its managed execution
environment, not a return to ad hoc venv/pip setup. Prefer uv commands and the
declared lockfiles over manual package installation into those environments.

Freezing dependencies and freezing an executable are separate capabilities.
When a distributable frozen Python controller is needed, build it with a
packager such as PyInstaller from the locked uv environment. Record any
separate worker runtime it still requires. Neither a lockfile nor a frozen
executable turns Python-dependent APIs into C++ implementations.

## Why Python must remain

- The adopted LoRA and embedding training approach uses `sd-scripts` and its
  PyTorch, Accelerate, Transformers, and Diffusers workflows. Those Python
  training APIs are not replaced by a C++ inference engine. This describes the
  training dependency decision; it does not claim that training has already
  been integrated or validated on Theory.
- The existing llama.cpp Hugging Face, tokenizer, and LoRA converters are Python
  programs with their own package and architecture support. A native converter
  for selected formats cannot replace unsupported architectures, tokenizers,
  or Python-dependent conversion behavior. See the current
  [Model Tools integration](../ui/easydiffusion/model_tools.py) and
  [native conversion plan](native-hf-lora-gguf-plan.md).
- Server plugins and helper integrations that require Python APIs must retain a
  Python execution path unless an equivalent implementation is actually
  available and validated. Moving REST transport or UI ownership into C++ does
  not, by itself, replace the functionality behind those interfaces.

LibTorch provides C++ tensor and autograd facilities, but it is not a drop-in
replacement for Python training scripts or the high-level APIs around them.
Likewise, bundling a Python controller with PyInstaller changes deployment; it
does not remove the interpreter or the worker's Python dependencies.

This does not mean every API is Python-only. HTTP service APIs, file formats,
and operations with suitable native runtimes can still have C++ implementations.
The limitation is preserving the complete set of required integrations, several
of which depend on Python-only APIs without practical C++ replacements.

## Development rules

1. Keep native inference and suitable compute paths in C++, while retaining
   Python for integrations that need it. Do not make elimination of Python a
   completion criterion for the project.
2. Maintain the application's supported Python environment as a lasting
   dependency, preferably managed with uv. Use explicitly configured,
   version-pinned, separately locked uv worker environments when training or
   another API requires a different Python/package combination.
3. Prefer subprocess or structured API boundaries for these workers. Tools such
   as uv are the preferred environment-management path; packaging must preserve
   their locked dependency and version requirements and support reproducible
   frozen deployments.
4. Track native and Python capability/readiness separately. Native support for
   one model family or operation does not establish coverage of all fallbacks.
5. Remove an individual Python package only after checking every consumer in its
   environment and validating any replacement. Keep packages needed by retained
   APIs, compatibility paths, training, plugins, or other model tools.
6. Describe native migrations as scoped improvements. Do not promise that
   sufficient progress on one subsystem will eventually remove Python from the
   rest of the application.
