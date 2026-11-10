# Theory CPU ownership and registry hardening

This follows the isolated CPU lifecycle import. It remains outside production
builds, inference and GPU runtime selection.

## Implemented

- CpuByteBuffer is non-copyable and nothrow movable. Move transfers the handle
  and empties the old owner. Reset/destruction releases through Common; repeated
  reset is harmless. Move assignment releases the previous owned buffer first.
- The facade exposes a const borrowed byte view, not the mutable raw ownership
  descriptor. The view expires on reset/move/destruction.
- Both load and unload registration tables use a mutex for registration and
  dispatch lookup. Each slot accepts only its first registration. Attempting to
  replace a registered handler fails instead of changing the release behavior
  of live resources.
- Callback execution occurs outside registry locks. Independent buffers can be
  owned by different threads. Same-object concurrent mutation remains prohibited.
- Original Theory hashes remain recorded. Explicit integrated_sha256 entries
  identify the two hardened Common registry files.

## Validation

Three tests pass with host C++17 and -Wall -Wextra -Werror:
1. Integrated-source provenance hash checks.
2. Existing 32-cycle typed lifecycle/unsupported-route fixture.
3. Compile-time rejection of copy ownership, move construction/assignment,
   self-move, repeated reset, invalid input and eight threads performing 1,024
   allocation/move/release cycles while attempting rejected handler replacement.

The same suite passes with ThreadSanitizer and with AddressSanitizer plus
UndefinedBehaviorSanitizer. LeakSanitizer cannot run under this executor's ptrace
environment; the ASan/UBSan pass explicitly uses ASAN_OPTIONS=detect_leaks=0.
Do not describe that pass as leak-check coverage.

Commands:
- python scripts/test_theory_cpu_lifecycle.py
- THEORY_TEST_SANITIZERS=thread python scripts/test_theory_cpu_lifecycle.py
- ASAN_OPTIONS=detect_leaks=0 THEORY_TEST_SANITIZERS=address,undefined python scripts/test_theory_cpu_lifecycle.py

## Remaining admission gates

- Raw low-level Resource descriptors are still copyable for upstream compatibility.
  Production callers must use the facade rather than manufacture/copy descriptors.
- Static registration sources must be retained by any eventual library/link target.
  Current fixtures compile them directly; static-archive link retention is not yet proven.
- Allocate/size limits, 32-bit overflow, module lifetime and stronger malformed
  backend-result validation remain to be designed before runtime admission.
- A model byte buffer is not a parsed model. No inference, GPU, UI or packaged
  application behavior is added or proven by this increment.
