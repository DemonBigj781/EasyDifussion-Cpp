# 15 — Wine / Proton Exclusive RAM Offloading

## Objective
Make the project's exclusive RAM/offloading mechanisms safe and useful when components run under Wine or Proton.

## Implementation
1. Document exactly what “exclusive RAM offloading” currently guarantees: reservation, pinning, file mapping, cache ownership, or eviction behavior.
2. Abstract OS memory operations behind one interface so native Linux, native Windows, and Wine/Proton behavior can differ safely.
3. Detect Wine/Proton from runtime/environment indicators and log the detected mode.
4. Test virtual allocation, mapped files, pinned host memory, large-page behavior if used, and process teardown under Wine.
5. Ensure memory accounting uses allocations actually owned by the process and never assumes exclusivity Wine cannot enforce.
6. Add conservative fallback to normal host allocation/offload when exclusive semantics are not reliable.
7. Test interaction with GPU drivers exposed through Wine/Proton.

## Dependencies
Windows memory abstraction and existing offload code need clear ownership semantics.

## Validation
Long-running repeated model load/unload, cancellation, process crash/restart, host memory pressure, and comparison of reported versus actual resident memory.

## Complete when
Wine/Proton mode can use offloading without leaks, corruption, false memory accounting, or system instability, with fallback when exclusivity cannot be guaranteed.
