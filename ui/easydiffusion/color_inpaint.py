"""Prepare separate inpaint and color influences for one diffusion call."""
from PIL import Image, ImageChops


def prepare_color_weighted_inpaint(original, guide, ordinary_mask, size, prompt_strength):
    if original is None:
        raise ValueError("Color-weighted inpaint requires an original image")
    if "A" not in guide.getbands():
        raise ValueError("Color guidance requires an alpha channel containing its weights")
    if guide.size != original.size:
        raise ValueError("Color guidance and the original image must have matching dimensions")
    if not 0 < prompt_strength < 1:
        raise ValueError("Color-weighted inpaint requires prompt strength between 0 and 1 (exclusive)")
    if guide.getchannel("A").getbbox() is None:
        raise ValueError("Color guidance contains no painted pixels")

    # Ordinary inpaint is the first influence. Keep its input unchanged.
    mask = ordinary_mask.convert("L").resize(size, Image.Resampling.BILINEAR) if ordinary_mask is not None else Image.new("L", size)
    base = original.convert("RGB").resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    weighted_guide = guide.convert("RGBA").resize(size, Image.Resampling.BILINEAR)

    # Advanced inpaint adds independent regions and alpha-weighted starting
    # colors. This is model conditioning, never a post-generation paint pass.
    guide_region = weighted_guide.getchannel("A").point(lambda value: 255 if value else 0)
    effective_mask = ImageChops.lighter(mask, guide_region).convert("RGB")
    conditioned = Image.alpha_composite(base, weighted_guide).convert("RGB")
    return conditioned, effective_mask
