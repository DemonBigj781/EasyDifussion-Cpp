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

Before production admission:
- Migrate all prospective consumers to the ownership facade. The low-level
  Theory Resource remains an internal-style raw handle and must not be copied
  by application consumers.
- Define a frozen registration phase for a production plugin system and explicit
  module lifetime; current immutable routes assume linked implementations live
  for the process lifetime.
- Define allocator/size limits and 32-bit behavior.
- Normalize backend result identity and error cleanup before adding plugins.
- Preserve foundation/runtime boundaries; do not replace current GGML or model
  ownership with this test-only prototype.
- Validate each additional typed resource and backend separately. Enum labels
  and empty directories are not implemented capabilities.

The unusual unload directory layout is retained verbatim for provenance.
