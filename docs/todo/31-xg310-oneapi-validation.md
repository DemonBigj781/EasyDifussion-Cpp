# 31 — H3C XG310 oneAPI / SYCL Validation

## Objective
Validate the H3C XG310 as a real Easy Diffusion oneAPI/SYCL compute target and separately document any cloud-gaming/rendering usefulness.

## Hardware model
Treat the board as multiple Intel SG1/Gen12 GPU devices with separate memory, not as one automatically unified 32 GB device. Device discovery must determine the actual runtime topology.

## Implementation / test sequence
1. Install a driver/oneAPI stack that enumerates the card reliably on the target Debian/Linux environment.
2. Capture `sycl-ls`/equivalent device identifiers, PCI topology, driver versions, global memory per device, max allocation, USM capabilities, subgroups, FP16 support, and relevant extensions.
3. Run allocation/copy/compute smoke tests on each device independently.
4. Build Easy Diffusion with its existing `ggml-sycl` backend enabled.
5. Validate components progressively: tensor ops → CLIP/text encoder → VAE → UNet/DiT → complete generation.
6. Test one GPU first, then multi-device scheduling. Do not depend on peer-to-peer until measured/proven.
7. Measure host-staged transfer bandwidth/latency and determine useful model/layer partition sizes.
8. Stress repeated generation, model switching, OOM cleanup, long runs, and thermal/power stability.
9. Cloud-gaming/rendering tests must be tracked separately; display/graphics success is not evidence that diffusion compute is correct.

## Related design
See `docs/easy-diffusion-oneapi-plan.md`.

## Complete when
The board's actual device/memory topology is documented, at least one end-to-end Easy Diffusion SYCL generation succeeds, multi-device limits are measured, and unsupported behaviors are clearly recorded.
