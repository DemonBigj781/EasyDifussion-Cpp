# Shared Plugin Manager

**C++ UI → Plugin Config** manages the same 26 catalog entries, including the
**Image Modifiers (show/hide)** switch, as the legacy **Plugins** tab.
It contains plugin switches and a search field, not
GPU/backend selection. GPU/backend controls remain in Settings and GPU Config.

Both interfaces use the catalog, preference handling and checkbox renderer in
`ui/media/js/local-plugin-preferences.js`. The existing browser storage key
`easy-diffusion-enabled-local-plugins-v1` and defaults-version migration are
preserved. Choices are shared between tabs in the same browser profile and
origin; they are not server-wide settings.

- Changes are saved automatically. Search matches plugin names and IDs.
- Turn off **Image Modifiers (show/hide)** to hide the modifier controls in
  both interfaces immediately, without reloading. Turn it on to restore them.
  Existing modifier selections are preserved and still apply to generation;
  this switch only changes visibility. Modifiers remain visible by default.
- In the legacy UI, enabling still uses the existing script loader. Disabling
  an already-loaded legacy plugin requires a page reload to remove its hooks.
- The C++ manager saves the same preferences without injecting legacy scripts
  into an incompatible native-page DOM. All currently listed catalog entries
  have modern-page implementations, including the 21 ports described below.
- Native Perchance page visibility follows the same switches. An open native
  page reloads when its preference changes, and a disabled page explains how
  to re-enable it.
- Plugin preferences never override kiosk restrictions. For example, enabling
  the Perchance image plugin does not make unrated generation available in kiosk
  mode.

Reload already-open pages after installing this UI update. Cross-tab preference
changes then synchronize through browser storage events.

## Modern generation plugins

Enable these in **Plugin Config**, then open **Main**. Their controls follow the
existing switches, including changes from another tab. Enabling support does
not automatically change which plugins you have enabled. Legacy implementations
and preference IDs remain intact.

- **Accessibility improvements:** hover/focus or mouse-button image actions;
  optional page context-menu suppression, with text fields preserved.
- **Animate:** video/GIF extraction and ordered image sequences, img2img
  rendering, GIF/WebM download or frames-only output.
- **Daily output folders:** uses the local calendar date (`YYYY-MM-DD`) as the
  session ID, reevaluated for each submitted task.
- **Disable source-image zoom:** controls hover zoom on source thumbnails.
- **GPU mode quick toggle:** low/balanced memory mode on render requests;
  does not select or reconfigure a GPU/backend.
- **Always-visible Make Image button:** a fixed Generate button while scrolling.
- **Processing-order quick toggle:** oldest-first or newest-first processing of
  waiting tasks; does not interrupt the active task.
- **Prompt diff:** escaped additions/deletions for both prompts on previews.
- **Prompt translator:** translates both prompts to English after explicit
  consent to send them to Google Translate; English sends nothing.
- **Queue counter:** remaining image/task counts in controls and browser title.
- **Rabbit Hole:** bounded parameter combinations, model selections, modifier
  alternatives, branching, step/guidance actions and img2img chains;
  gallery/classic display and selection tools.
- **Random-seed quick toggle:** switches between random and fixed seeds.
- **Batch seed randomizer:** normal, prompt-based CRC32,
  positive/negative-prompt CRC32, or fully random seeds.

The **Generation plugins** panel contains the smaller controls. **Rabbit Hole**
and **Animate** have expandable panels beneath the regular generation settings.
These are browser-side ports integrated with the existing C++ page renderer,
not a conversion of the plugin algorithms into compiled C++.

### Queue and output behavior

- Generate may enqueue additional tasks while a task is running. Prompt syntax
  such as `a {red,blue} car` creates separate requests; the combined positive/
  negative prompt expansion is limited to 256 variants.
- The queue runs one task at a time and holds at most 512 waiting tasks. Stop
  cancels waiting tasks and requests cancellation of the active backend task.
- Daily folders require server-side saving to be enabled, and the server's
  folder format must include `$id` (the default). The plugin does not override
  your save directory or custom folder-format configuration.
- Translation requires selecting a language and checking **Allow sending
  prompts to Google Translate**. Consent is separate from the plugin switch.
  Network/translation failures are reported rather than silently rendering
  untranslated text. No Google requests are made by the regression suite.

### Animate and Rabbit Hole

Animate uses the current checkpoint, prompt, dimensions and seed. Choose one
video, one GIF, or multiple images in frame order. Set the prompt strength and
FPS; video start/end times select the input range. Each frame is submitted only
after the preceding frame completes. GIF and WebM exports are encoded locally
in the browser using the rendered frames. The GIF encoder/decoder are reused
from the bundled legacy Animate libraries, without loading the legacy UI hooks.

Animations are limited to 120 frames and 64 million output pixels in total;
reduce the frame count or dimensions if that limit is exceeded. Video decoding
depends on browser codec support, and WebM export requires MediaRecorder and
canvas capture support. **Stop animation and queue** also clears other waiting
tasks on that page. Disabling Animate prevents further frames from being added;
use Stop to cancel the currently submitted task.

Rabbit Hole generates combinations of the chosen ranges, capped by **Maximum
images** (1–512). A count of 1 uses the specified midpoint. Unselected model
lists retain the current model. Select multiple entries to compare checkpoints,
VAEs, LoRAs, samplers or face-correction models. Modifier alternatives are one
per line. Model choices must still be compatible with each other and the
backend; a plugin switch does not bypass kiosk restrictions.

Each generated image offers branching, configurable step/guidance increments,
and an img2img chain that uses each preceding result as its next source. The
native controls preserve the modern page layout rather than reproducing Rabbit
Hole's legacy full-page CSS. Modern Rabbit Hole settings have their own storage
key; legacy Rabbit Hole settings are not overwritten or automatically migrated.

## Additional image and prompt tools

These eight switches reveal panels or image actions on **Main** without loading
their legacy page hooks:

- **Spell tokenizer and merged tag search:** comma-token editing, drag/arrow
  reordering, removal, duplicate highlighting, Alt + wheel emphasis, and token
  estimates. A dedicated tag-search input queries the existing merged-tag
  worker and aliases without competing with Prompt assistance's typing hooks.
  The worker starts only when a search is requested and stops when disabled.
  The bundled legacy SFW list is available and is mandatory in kiosk mode.
- **Stig image-to-img2img:** upload a source or choose **Use as img2img source**
  on a generated image. Generate can use that source, or run a bounded sequence
  with output-to-input feedback, separate adjustment/reset intervals for
  guidance, strength, LoRA weights and seeds, and periodic zoom/rotation/pan.
  **Zero all adjustments** disables the increments and transforms.
- **Stig image utilities:** rotate left/right, tile horizontally/vertically or
  2 × 2, inspect request information, and download the transformed PNG with a
  JSON metadata sidecar. These edit browser previews, not saved originals.
  The legacy plugin's unimplemented mask actions are not exposed.
- **Stig LoRA shuttle controls:** image actions set weights to 0, 0.5 or 1,
  shift by ±0.1, or run the 11-point 0–1 grid. Multi-LoRA arrays are preserved;
  shuttle weights are clamped to the legacy 0–1 interval.
- **Stig text-to-prompt:** import or paste one prompt per line, search, navigate
  first/previous/next/last/random, add pre/post text, or generate selected,
  random or sequential batches. Duplicate lines are retained. Empty lines are
  ignored and the final line need not have a newline.
- **Storyteller:** the Main workspace gains Generate/Storyteller tabs. Use
  **Add to Storyteller** on images, then save/remove individual images or all
  board images. Like the legacy board, it is in-memory and clears on reload.
- **Template manager:** capture settings or an image request, save, load/edit,
  rename, delete, filter, run, and import/export JSON backups. It uses the
  existing `EasyDiffusionSettingsDatabase` / `EasyDiffusionSettings` database
  and the `task templates` record, retaining the legacy record shape and
  unknown metadata fields. Imports preserve existing same-name records.
- **Browser spellcheck toggle:** changes spelling assistance on positive,
  negative and editable modifier/wildcard fields using the existing
  `enable_spellcheck` preference. Disabling the plugin restores prior attributes.
  While enabled, it takes precedence over Prompt assistance's spellcheck option;
  disabling it returns control to Prompt assistance if that plugin is active.

### Tool integration and limits

Img2img sequences can use **Stig wildcard — sequential/random** as their prompt
source. Enable Stig text-to-prompt and load its lines first. **Change wildcard
prompt every N frames** selects the interval. The original optional Build A
Scene and Dress Me Up integrations are not included; those separate plugins
are not in this catalog. Source images stay in memory rather than browser
storage. Disabling the sequence plugin prevents subsequent tasks; **Stop
sequence and queue** also requests cancellation of the active task.

Wildcard and template batches, and img2img sequences, are capped at 512 tasks.
Wildcard imports are limited to 2 MB; template backups to 16 MB; uploaded
img2img sources to 32 MB. Transformed image previews are capped at 32 million
pixels to avoid runaway repeated tiling.

Templates load into an editable JSON request panel. **Run template** reproduces
the complete stored request, while **Use template prompts in generator** copies
only the positive/negative prompts. **Restore template seeds** is optional;
otherwise runs use fresh random seeds. Every run receives the current session
ID rather than the saved session ID. Server model validation and kiosk rules
still apply. Writes use one read/modify/write IndexedDB transaction, so template
operations do not replace the list with a stale copy from another modern tab.

Regression checks:

`node scripts/test_local_plugin_preferences.cjs`

`node scripts/test_cpp_plugin_manager_ui.cjs`

`node --test scripts/test_cpp_optional_plugins.cjs scripts/test_cpp_creative_plugins.cjs`

`node --test scripts/test_cpp_workbench.cjs`

`node scripts/test_cpp_optional_plugins_ui.cjs`

The new browser suite invokes the compiled page renderer and intercepts every
HTTP request. It checks queue ordering/cancellation, request options, translation
consent/failure, safe prompt diffs, Rabbit Hole combinations/chains, both GIF and
WebM input/output paths, cross-tab preferences, and desktop/mobile layout.
Its workbench checks cover merged aliases, token editing, wildcard batches,
LoRA arrays/grid, image transforms/metadata, Storyteller, scheduled feedback,
legacy-compatible template storage/import/export, and browser spellcheck.
It does not run GPU inference or change backend configuration. Set
`PLAYWRIGHT_MODULE`, `CHROMIUM_PATH`, or `CPP_UI_RENDERER` when those tools are
outside their default locations.

The plugin-manager browser test uses an isolated browser context and the actual
legacy manager functions in a minimal DOM fixture. It checks both directions,
the legacy loading callback, persistence, filtering, native-page gating, kiosk
priority and mobile layout without changing backend configuration or starting
generation. It does not bypass the live kiosk restriction on `/legacy`.
