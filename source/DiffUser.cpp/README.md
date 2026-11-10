# Experimental Theory CPU resource contract

This began as a selective source import from Theory commit
4755b7abe48b49207e33bbff786dc3e2a7032491. It remains outside production inference. An opt-in
SDKIT_BUILD_THEORY_CPU_RESOURCES target now tests the facade against the existing
GGML CPU allocator; default builds are unchanged. See
../../docs/integration/theory-lifecycle-source.json for the original source hashes and explicit integrated-file overrides.

The admitted slice covers Common dispatch keyed by backend and resource type,
plus CPU allocation/copy/release of arbitrary byte buffers. A successful load
does not validate a model, initialize a tensor, or prove inference. Only the CPU
model-byte route is registered; VAE, image, mask, latent and other backend routes
remain unsupported.

Run python scripts/test_theory_cpu_lifecycle.py from the repository root.
It compiles these sources using the host C++ compiler and runs synthetic-byte
fixtures. No models, drivers, GPU calls or network access are involved.

The new api/cpu_byte_buffer.hpp facade provides exclusive move-only ownership.
It releases automatically and exposes only a borrowed const byte view. Independent
owners can run concurrently; the same owner must not be mutated concurrently.
Common registration and lookup are mutex-protected, and registered routes cannot
be replaced. Backend callbacks run outside registry locks.

The public header now uses opaque private state. Common dispatch retains CPU
registration explicitly when linked from a static archive. Byte allocations
have a configurable 256 MiB default budget and addressability checks. Malformed
load results are rolled back through the originating translation, never by
guessing an allocator in Common.

Before production admission:
- Migrate consumers to the facade; raw Common descriptors remain internal-style
  contracts, not safe copyable ownership objects for application callers.
- Dynamic callbacks must supply a module guard; immutable registrations retain
  it. Hot unload is not supported. Real plugin admission remains unimplemented.
- Translations must publish provisional ownership before throwing. Native CPU
  allocation identities reject stale/foreign unloads; every additional provider
  must establish equivalent truthful ownership semantics.
- Preserve foundation/runtime boundaries and prove actual inference-resource,
  model and backend integration separately.

See ../../docs/integration/theory-runtime-gates-2026-11-10.md for evidence and
remaining boundaries. The unusual unload layout remains for source provenance.

See ../../docs/integration/theory-ggml-cpu-2026-11-10.md for the real CPU backend-buffer
execution evidence and its limits. No model inference is claimed.
