# Resource reservation feature scaffold

Status: **future-only scaffold**.

This directory is reserved for a future normalized contract that protects
system resources for DiffUser.cpp. It currently reserves no RAM, VRAM, devices,
threads, or disk space and has no build references or support claim.

The contract must be designed alongside the memory oversubscription plans. It
may provide safety margins or reservation primitives to Overflow, but it must
not duplicate Overflow's cross-tier selection, allocation, release, or migration
policy. See [the layout audit](../../LAYOUT_AUDIT.md) and the
[memory oversubscription plan](../../../../docs/todo/33-memory-oversubscription-handler.md).
