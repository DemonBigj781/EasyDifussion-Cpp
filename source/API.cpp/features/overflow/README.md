# Overflow

Overflow owns the normalized policy used when an allocation no longer fits in
the active memory tier. Its CUDA implementation is an application-level Linux
replica of NVIDIA's Windows CUDA system-memory fallback behavior; it does not
install, replace, or reconfigure a display driver.

## Allocation order

The common planner preserves this order:

1. active-device VRAM;
2. free VRAM on another runtime-usable GPU, ordered by available capacity;
3. `gpu-zram`, when its backend exists;
4. system RAM that remains GPU-accessible;
5. host zram;
6. disk-backed swap.

Candidates carry both a backend and device index. This is required for mixed
systems: CUDA can allocate on another NVIDIA CUDA device, but an AMD or Intel
device must be executed through its ROCm, OpenCL, Vulkan, or oneAPI translation.
Common owns the order; each backend translation owns only allocations that its
native runtime can prove usable.

## CUDA system-memory replica

The implemented CUDA path is:

```text
cudaMalloc(active GPU)
    -> cudaMalloc(secondary CUDA GPU)
    -> cudaMallocManaged(CPU-preferred, active GPU accessed-by)
    -> cudaHostAllocMapped
```

`cudaMallocManaged` is used only when the active device reports both Managed
Memory and concurrent managed access. The allocation is advised to prefer CPU
residency and to remain accessible by the active GPU. If that capability or
allocation is unavailable, pinned mapped host memory provides the final
GPU-accessible RAM tier. HMM/ATS can improve unified addressing when present,
but it is not required by this route.

Every resource records its actual allocator and owner. Device allocations use
`cudaFree`, managed allocations use `cudaFree`, mapped allocations use
`cudaFreeHost`, and CPU heap allocations use `free`. An allocation is not
cleared until its owning release operation succeeds.

The host-memory reserve is checked before either system-memory fallback. This
prevents an attempted GPU recovery from exhausting the operating system's RAM.
`gpu-zram`, zram, swap, and non-CUDA GPU executors remain planned integrations;
their presence in the common plan does not claim runtime support.

## Hardware implications

- Ampere devices such as RTX 3060/3060 Mobile use Managed Memory when the
  runtime reports concurrent access.
- Volta V100 follows the same capability query and retains mapped host memory
  as a fallback.
- Maxwell devices such as Tesla M40 generally contribute secondary CUDA VRAM
  but use mapped host memory when concurrent Managed Memory is unavailable.
- Kepler-era devices such as Quadro K2000 and pre-GTX hardware are accepted
  only if an installed backend actually enumerates and allocates on them.
- AMD cards, including RX 580 2048SP, require a working non-CUDA translation.
- H3C XG310 is four independent low-power Intel SG1 GPUs with 8 GB each. The
  planned oneAPI/OpenCL validation must expose four candidates rather than one
  contiguous 32 GB allocation.

These are selection rules, not hardcoded product allowlists. Runtime capability
and successful native allocation remain authoritative.

## Validation

From `source/API.test`:

```text
make run-overflow-cpu
make run-overflow-cuda
make run-overflow-cuda-stress
```

The regular CUDA test forces the system-RAM tier, launches a CUDA kernel through
the returned pointer, verifies the touched pages on the CPU, and releases the
resource. The opt-in stress test reserves VRAM while retaining a 1536 MiB GPU
safety margin, makes a 2048 MiB `cudaMalloc` fail, requires the normalized
result to record that real OOM, verifies GPU access to the fallback allocation,
then releases both the fallback and all reservations. It also retains at least
8 GiB of available host RAM.
