# 19 — Stable Horde Client API / UI

## Objective
Add Stable Horde as an optional remote generation client with complete job lifecycle handling.

## Implementation
1. Create a provider client covering authentication, capability/model discovery, generation submission, queue/status polling, cancellation, and result retrieval.
2. Define a provider-neutral remote-job model distinct from local sdkit jobs while exposing a consistent UI lifecycle.
3. Translate only supported generation parameters and report ignored/unsupported fields before submission.
4. Add model/worker capability display where the API provides it.
5. Store credentials through the project's settings/secret mechanism; never commit keys.
6. Add progress/queue state, estimated wait when available, cancellation, retries with bounded backoff, and result metadata.
7. Keep client mode separate from any future worker mode; a client should not unexpectedly advertise local hardware.

## Failure behavior
Remote API/network failures must not affect local generation. Preserve detailed provider error messages without leaking credentials.

## Validation
Anonymous/authenticated flow where supported, submit, poll, cancel, timeout, provider error, result download, duplicate-click protection, restart/resume policy.

## Complete when
A user can submit, monitor, cancel, and retrieve a Stable Horde generation entirely from Easy Diffusion's UI/API.
