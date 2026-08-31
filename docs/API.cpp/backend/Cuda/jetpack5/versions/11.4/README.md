# JetPack 5 — CUDA 11.4

CUDA 11.4 is the default CUDA generation for JetPack 5.x. This directory tracks API.cpp support for the JetPack 5 platform variant at CUDA 11.4.

## Required API.cpp feature coverage

- Attention: FlashAttention, SageAttention, xFormers, FlexAttention
- Cache: EasyCache, TeaCache

Each feature page must record compile status, Jetson GPU architecture support, native/adapted/fallback implementation, datatype and kernel restrictions, and runtime validation.