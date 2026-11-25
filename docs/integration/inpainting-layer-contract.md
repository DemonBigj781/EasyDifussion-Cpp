# Inpainting layer contract

The original source image remains immutable. Image-edit drawings are composited onto a temporary working image before inpainting preparation.

Basic inpainting enables editing without a color preference. Advanced inpainting enables editing with an optional user-painted color preference. Both contribute to one generation operation, not separate diffusion passes.

## Preparation order

1. Composite the image-edit drawing overlay onto a temporary copy of the original.
2. Prepare the basic edit mask, applying optional inversion exclusively to this layer.
3. Incorporate advanced painted regions outside the effective basic mask, retaining their color preferences.
4. Combine edit eligibility and color guidance for diffusion processing.

Unmarked regions receive no explicit editing encouragement. Painted colors are preferences, not mandatory output pixel values. Basic regions have no explicit color preference. Never invert the combined mask or the advanced color map.

## Acceptance criteria

- Original image remains unchanged.
- Basic-only editing has no explicit color preference.
- Advanced-only painted regions enable editing with their painted color preference.
- Combined basic and advanced regions participate in one generation.
- Basic inversion does not invert or discard advanced color data.
- No supplied mask preserves ordinary img2img behavior.
- ControlNet and segmentation inputs are unaffected.

This is the intended contract, not a claim of current runtime parity. The recovered native inversion currently consumes the supplied mask; upstream composition must be audited before enabling inversion on combined inputs.
