# 22 — MCP API

## Objective
Expose a safe, stable subset of Easy Diffusion capabilities through a Model Context Protocol server/tool interface.

## Implementation
1. Start with bounded tools: `list_models`, `get_capabilities`, `generate`, `get_job`, `cancel_job`, and `get_result`.
2. Define explicit JSON schemas independent of internal Python/C++ structures so backend refactors do not break clients.
3. Route generation through the existing authoritative server/job system rather than allowing MCP to call model code directly.
4. Add resource identifiers for generated artifacts instead of arbitrary filesystem paths.
5. Restrict bind address/authentication appropriately; default local-only unless remote access is explicitly configured.
6. Do not expose arbitrary shell commands, unrestricted file reads/writes, or raw process control.
7. Include backend/device/model capability information so an MCP client can make valid requests.
8. Add protocol/version logging and robust cancellation/disconnect cleanup.

## Dependencies
Stable job API and model capability schema.

## Validation
Schema discovery, generation lifecycle, cancellation, invalid input, concurrent requests, client disconnect, artifact retrieval, auth restrictions, and server restart.

## Complete when
A standard MCP client can discover and safely execute the supported Easy Diffusion workflow without relying on undocumented internals.
