def _combine_prompt_for_clip(visible_prompt: str, hidden_embedding_prompt: str) -> str:
    visible = (visible_prompt or "").strip()
    hidden = (hidden_embedding_prompt or "").strip()

    if not hidden:
        return visible
    if not visible:
        return hidden

    separator = " " if visible.endswith(",") else ", "
    return f"{visible}{separator}{hidden}"


def combine_positive_prompt_for_clip(prompt: str, hidden_positive_prompt: str) -> str:
    """Append embedding-only positive text at the CLIP encoding boundary."""
    return _combine_prompt_for_clip(prompt, hidden_positive_prompt)


def combine_negative_prompt_for_clip(negative_prompt: str, hidden_negative_prompt: str) -> str:
    """Append embedding-only negative text at the CLIP encoding boundary."""
    return _combine_prompt_for_clip(negative_prompt, hidden_negative_prompt)
