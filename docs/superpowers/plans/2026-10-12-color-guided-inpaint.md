# Color Guided Inpaint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate color-guidance canvas for inpainting, while keeping the normal inpaint mask and object-removal mask pipeline unchanged.

**Architecture:** Reuse the existing image editor as a color-guide paint surface, but save its output into a separate `window.inpaintColorGuidanceImage` data URL instead of `mask`. Native/image-loader preview composes the base image, normal mask overlay, and optional color-guidance overlay. A `PLUGINS.TASK_BUILD` hook sends `inpaint_color_guidance` only when guidance exists.

**Tech Stack:** Existing browser JavaScript plugins, `ImageEditor`, `PLUGINS.TASK_BUILD`, Python unittest static integration tests.

---

### Task 1: UI contract tests

**Files:**
- Modify: `scripts/test_ui_plugin_integration.py`

- [ ] Assert the inpaint plugin exposes `init_image_button_color_guide`, `window.inpaintColorGuidanceImage`, and `inpaint_color_guidance`.
- [ ] Assert native image tools preview includes the color guidance layer.

### Task 2: Separate color guidance editor

**Files:**
- Modify: `ui/plugins/ui/inpaint_plugin/inpainting.plugin.js`

- [ ] Add a Color Guide button near Draw/Inpaint controls.
- [ ] Open the existing image editor as a color painting surface initialized from the current preview.
- [ ] Override save only for that flow so the painted color/alpha image is stored as `window.inpaintColorGuidanceImage`.
- [ ] Add a `PLUGINS.TASK_BUILD` hook that sets `reqBody.inpaint_color_guidance` when guidance exists.

### Task 3: Preview integration

**Files:**
- Modify: `ui/plugins/ui/image_plugin/native-image-tools.plugin.js`

- [ ] Keep Text remove mask, Object detect mask, Face mask, Preview mask, and Remove masked object separate.
- [ ] Make Preview mask include `window.inpaintColorGuidanceImage` as a colored overlay when present.

### Task 4: Verification

**Commands:**
- `python -m py_compile scripts/test_ui_plugin_integration.py scripts/test_native_image_tools.py`
- `python -m unittest scripts.test_ui_plugin_integration.TestBundledUIPluginIntegration.test_native_image_tools_keep_mask_creators_and_mask_consumer_separate scripts.test_ui_plugin_integration.TestBundledUIPluginIntegration.test_inpaint_editor_exposes_invert_selection_action scripts.test_ui_plugin_integration.TestBundledUIPluginIntegration.test_draw_open_preserves_preview_source_when_dimensions_are_pending scripts.test_ui_plugin_integration.TestBundledUIPluginIntegration.test_color_guided_inpaint_uses_separate_guidance_image`
- `python scripts/test_native_image_tools.py`
- `node --check ui/plugins/ui/image_plugin/native-image-tools.plugin.js`
- `node --check ui/plugins/ui/inpaint_plugin/inpainting.plugin.js`
- `git diff --check -- scripts/test_ui_plugin_integration.py scripts/test_native_image_tools.py ui/plugins/ui/image_plugin/native-image-tools.plugin.js ui/plugins/ui/inpaint_plugin/inpainting.plugin.js ui/easydiffusion/native_image_tools.py ui/plugins/server/native_image_tools/native_image_tools.py docs/superpowers/plans/2026-10-12-color-guided-inpaint.md`
