# CPU xFormers definitions

This directory owns the CPU-native request, validation, staged attention
operations, and fused forward sequencing. It does not include or register with
Common.

The matching translation under `../translation/cpu/` converts the normalized
request and score buffers, invokes these definitions, converts results, and
registers the route with Common. Compilation and runtime revalidation remain
pending after this ownership correction.
