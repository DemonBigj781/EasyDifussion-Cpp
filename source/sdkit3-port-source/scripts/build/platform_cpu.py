"""
CPU platform build configuration.

IMPORTANT: By default, GGML_NATIVE is disabled to ensure binaries are portable
across different Intel processors. When GGML_NATIVE=ON, the build uses -march=native
which optimizes for the build machine's CPU, causing crashes on CPUs without the
same instruction sets (especially when using offload_to_cpu feature).

The current configuration enables AVX2/FMA/F16C/BMI2 which provides good performance
while maintaining compatibility with Intel CPUs from ~2013 onwards (Haswell+).

If you need different optimization levels:
- For maximum compatibility: Remove AVX2/FMA/F16C/BMI2 flags (slower)
- For newer CPUs only: Add AVX512 flags (faster, less compatible)
- For local builds only: Set GGML_NATIVE=ON (fastest, not portable)
"""


def check_environment():
    """CPU build always available."""
    print("CPU build environment ready.")
    return True


def get_compile_flags(target_any):
    """Get compile flags for CPU."""
    # Disable GGML_NATIVE to make binaries portable across different Intel CPUs
    # This prevents -march=native which would optimize for the build machine only
    # Enable AVX2 + FMA + F16C which are supported by most Intel CPUs from ~2013+
    # You can adjust these based on your minimum supported CPU
    return ["-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_BMI2=ON"]


def get_platform_name():
    """Get platform name for CPU."""
    return "cpu"
