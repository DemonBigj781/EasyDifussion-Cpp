# GGUF feature scaffold

Status: **future-only scaffold**.

This directory reserves a possible normalized DiffUser.cpp boundary for later GGUF
work. It has no method contract, build references, parser, writer, converter, or
support claim.

The existing [native conversion plan](../../../../docs/native-hf-lora-gguf-plan.md)
currently proposes the native converter under
`source/sdkit3-port-source/src/conversion/`. Decide whether this feature will
own format I/O, conversion orchestration, or another narrow contract before
adding source. See [the dated layout audit](../../../../Audit/2026-09-06/LAYOUT_AUDIT.md).
