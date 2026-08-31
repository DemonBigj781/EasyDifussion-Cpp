# 20 — Complete Civitai / Hugging Face API and UI

## Objective
Finish provider-integrated model discovery, metadata, download, validation, and model installation for Civitai and Hugging Face.

## Implementation
1. Define a provider-neutral model-source interface: search, pagination, model metadata, versions, files, authentication, download URL, license, and requirements.
2. Implement Civitai and Hugging Face adapters independently so provider-specific fields do not leak through the UI.
3. Add filters, pagination, model/version/file selection, preview metadata, license display, base-model compatibility, and file size/hash.
4. Add resumable downloads, progress, cancellation, checksum verification, atomic final rename, and corrupted-partial cleanup.
5. Support authenticated/gated/private HF repositories with clear authorization errors.
6. Classify downloaded artifacts by model type and route them to the correct model directory; never infer type only from `.safetensors`.
7. Save source URL/provider/model/version/hash metadata for update checking and provenance.
8. Integrate native HF→GGUF conversion as an optional post-download action when applicable.

## Validation
Search/paging, auth/no-auth, interrupted resume, checksum mismatch, gated repo, version switching, model placement, duplicate download, and immediate model discovery after install.

## Complete when
Users can discover, inspect, download, validate, install, and use supported Civitai/HF assets without manually moving files.
