# First selective Theory slice

Source: progress/theory-mint-2026-11-10 at
4755b7abe48b49207e33bbff786dc3e2a7032491.

## Imported

Eleven files from Common load/unload plus CPU definitions/translations are
preserved byte-for-byte under source/DiffUser.cpp. The manifest records SHA-256
for every imported file. This is an isolated experimental source slice, not
linked into SDKit, GGML, Cosmopolitan, or either UI.

The useful contract is dispatch keyed by both backend and resource type.
Unsupported VAE/image/mask/latent routes cannot silently use the model route.
Its CPU implementation copies arbitrary bytes into malloc storage and frees
them later. The term model resource does not imply parsing weights or inference.

## Evidence

- Host C++17 compile passed with -Wall -Wextra -Werror.
- A synthetic six-byte payload passed 32 load/verify/unload cycles.
- Empty source, unsupported/invalid backend, unsupported resource kinds,
  repeated unload, and wrong-type unload rejection passed.
- All eleven source hashes match the pinned preservation commit.
- Both Python tests passed; all 117 foundation tests still pass.
- No model, GPU, external service or application execution occurred.

## Explicitly deferred

The broader Theory tree is not ready to replace the active runtime. Its
FULL_COMMON_CONTRACT_REAUDIT documents ownership violations in attention
translation layers, incomplete primitive routes and generation planning
without model execution. Those audit claims are not themselves current runtime
proof, but reviewed load/unload sources do confirm that the imported slice is
only byte ownership.

Before enabling this code in production, review move-only/resource identity
semantics, allocator limits, registration concurrency, backend-result identity
and cleanup on malformed translation results. Next safe increment is to
formalize and test those admission boundaries. Do not import unused backend
directories or advertise a backend because it appears in an enum.
