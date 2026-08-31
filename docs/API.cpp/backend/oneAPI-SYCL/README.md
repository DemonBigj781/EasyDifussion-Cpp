# oneAPI / SYCL backend

## Backend status

**Present, validation incomplete.**

The stable-diffusion.cpp tree contains a `ggml-sycl` backend and the repository contains dedicated oneAPI/SYCL planning and validation work. This backend therefore must be tracked as a real API.cpp target rather than a hypothetical backend.

Current documentation policy is conservative: a generic SYCL backend existing does not prove that each optimized attention family has a native SYCL implementation.

## Attention

See `attention/` for FlashAttention, SageAttention, xFormers, and FlexAttention status.

## Cache

EasyCache and TeaCache are backend-neutral host-resident policies. Their policy/state does not currently become a SYCL kernel merely because model execution uses SYCL.

## Validation

Hardware/runtime validation remains required, including the repository's XG310 oneAPI validation work.