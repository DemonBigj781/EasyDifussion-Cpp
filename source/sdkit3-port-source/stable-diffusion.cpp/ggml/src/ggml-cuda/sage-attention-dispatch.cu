// Keep the backend-neutral SageAttention dispatcher at the top level so both
// ggml-cuda and ggml-hip pick it up through their existing *.cu source globs.
#include "sage/sage-attention.cu"
