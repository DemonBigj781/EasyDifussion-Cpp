"""Validated request parameters for the native UNet sampling optimizations."""
import math


def native_sampling_options(tome_ratio=0.0, negative_guidance_min_sigma=0.0):
    ratio, sigma = float(tome_ratio), float(negative_guidance_min_sigma)
    if not math.isfinite(ratio) or not 0 <= ratio <= 0.5:
        raise ValueError("Token merging ratio must be between 0 and 0.5")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Negative guidance minimum sigma must be finite and non-negative")
    return {"tome_ratio": ratio, "negative_guidance_min_sigma": sigma}
