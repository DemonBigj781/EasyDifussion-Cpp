# Experimental Theory CPU resource contract

This is a selective source import from Theory commit
4755b7abe48b49207e33bbff786dc3e2a7032491. It is deliberately **not connected**
to the application build or any inference backend. See
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
- Define module lifetime and plugin admission. Registered implementations and
  rollback callbacks currently live for the process lifetime.
- Require allocator-valid pointers, strong exception safety and truthful unload
  status from every translation.
- Preserve foundation/runtime boundaries and prove actual inference-resource,
  model and backend integration separately.

See ../../docs/integration/theory-runtime-gates-2026-11-10.md for evidence and
remaining boundaries. The unusual unload layout remains for source provenance.
