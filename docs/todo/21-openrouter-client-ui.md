# 21 — OpenRouter Client UI

## Objective
Provide optional OpenRouter-backed text/LLM functionality for prompt assistance and other language tasks within Easy Diffusion.

## Implementation
1. Add provider settings for API key, base endpoint, preferred model, timeout, and optional generation defaults.
2. Implement model-list retrieval/cache where available and manual model ID entry as fallback.
3. Add standard and streaming chat/completion requests with cancellation.
4. Define specific Easy Diffusion actions such as prompt expansion, prompt rewrite, negative-prompt suggestions, metadata/caption assistance, rather than exposing an unstructured dependency throughout the app.
5. Keep local image generation independent of network/provider availability.
6. Surface model/context limitations and provider errors clearly; avoid silently truncating prompts.
7. Store credentials securely through settings and redact them from logs.
8. Add provider abstraction so a future local or alternate LLM service can implement the same actions.

## Validation
Key validation, model listing, streaming, cancellation, timeout, invalid model, context overflow, rate limit, offline behavior, and action integration.

## Complete when
A configured user can invoke at least one well-defined Easy Diffusion text-assistance action through OpenRouter with correct streaming/error handling.
