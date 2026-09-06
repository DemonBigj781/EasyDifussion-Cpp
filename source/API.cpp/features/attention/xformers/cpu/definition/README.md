# CPU xFormers definitions

This is a structure-only placeholder for a future CPU-native definition layer.
Definitions will describe native CPU capability and request semantics; they will
not perform Common translation.

No functional definition is implemented here yet. The current registered CPU
implementation is active under `../translation/cpu/` and talks directly to the
normalized Common contract. This missing intermediate definition layer is
layout debt, not evidence that the CPU route is unimplemented.
