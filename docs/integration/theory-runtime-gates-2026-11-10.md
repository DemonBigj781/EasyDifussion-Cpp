# Theory CPU archive, allocation and result gates

This increment remains an isolated CPU byte-storage component. No production
inference or GPU backend is enabled.

## Static archive retention

Common dispatch now explicitly references the CPU registration functions.
Their function-local initialization is thread-safe. Ordinary archive linking
therefore retains required translation objects without whole-archive flags or
forced undefined symbols. A fixture includes only the public facade header and
links a static archive with section garbage collection, then allocates, moves
and releases buffers.

## Ownership surface and allocation budgets

The public facade now contains an opaque private state. Its header no longer
includes or exposes Common raw-handle descriptor types. The const byte view is
borrowed and still expires on move, reset or destruction.

Requests have a default 256 MiB byte budget, explicitly overridable by a caller
with a reviewed budget. Common rejects empty/over-budget requests before invoking
a backend. The CPU definition also checks size_t and ptrdiff_t addressability
before allocation or copying. Tests cover zero budget, insufficient budget and
maximum-size overflow using tiny synthetic inputs without allocating huge buffers.

The private owner is allocated before backend storage, preventing owner allocation
failure from orphaning an existing byte buffer.

## Malformed load results

Registration requires both a load translation and its noexcept rollback callback.
Common checks success, handle presence, exact requested backend/resource type/
size, device identity and allocation budget. On a malformed result containing a
handle, only the originating translation's rollback function releases it; Common
never guesses the allocator. The CPU rollback invokes its native definition's
release function.

Six synthetic malformed-result cases (false success status, wrong backend, wrong
resource type, wrong size, wrong device and zero size) verify one rollback per
allocation and no surviving handle. The test's fake Mesa enum slot uses ordinary
malloc memory only; it is not a Mesa implementation or GPU validation.

## Verification

Five tests pass normally, with ThreadSanitizer, and with ASan/UBSan using
ASAN_OPTIONS=detect_leaks=0. LeakSanitizer remains incompatible with this ptrace
executor. The explicit malformed-result allocation/rollback counts are targeted
cleanup evidence, not general leak-scanner coverage.

The tests include the earlier eight-thread/1,024-cycle ownership stress and
32-cycle typed lifecycle fixture, plus static archive and malformed-result cases.
No large model, GPU, Media drive or video LoRA was used.

## Remaining boundaries

- Existing raw Common descriptors remain available to internal tests; production
  consumers must use the facade.
- Translation code must return valid allocator-owned pointers and provide strong
  exception safety before returning ownership. A validator cannot make an
  arbitrary pointer safe or recover a handle hidden by a throwing implementation.
- Unload callbacks must truthfully report whether ownership was released.
  Incorrect unload status after freeing cannot be safely undone. The owner
  terminates on that invariant failure rather than silently reusing the handle.
- Module unload, integration with actual inference resources and shared runtime
  allocators, platform packaging and real model/backend behavior remain unproven.
