# Pages

Store page definitions and page-level composition here. `src/pages.cpp` holds
the navigation catalog and C++ page builders. Pages assemble shared components from
`../Shared_Core` and feature-owned components, while leaving reusable
primitives in their owning modules.

The catalog provides Main, Canvas, Perchance Text/Image/Gallery, Image Gallery,
Dataset Gallery, Tagging, Training, Model Downloading, Settings, GPU Config,
Plugin Config, Stats, Logs, and Console destinations. Perchance and Galleries
appear as grouped primary navigation menus. System pages are in a separate
navigation row. The FastAPI host serves the C++ Main page at `/` when the
renderer is available;
other C++ pages use the `/cpp-ui` namespace. The page builders provide the
navigation and page structure. Main generation is split into `Generate/Main.cpp`
and feature modules under `Generate/`; its browser bridge loads checkpoints,
submits `/render` or `/video`, handles stop/progress, and displays returned images. LoRA and
ControlNet panels contribute request options through the Generate plugin
injection point. Canvas editing is provided by `plugin_scripts/canvas_workspace.js`
and is available independently of optional legacy inpainting enhancements.
Other page interactions are being connected incrementally.
Render Settings includes opt-in native spatial token merging and negative-guidance
minimum sigma. Both default to zero; see `../../../docs/native-sampling-optimizations.md`
for supported models, restrictions, and measured tradeoffs.
Dataset Gallery keeps processed images beside the selected-for-dataset column
and reserves the hidden pre-filter area for duplicate detection.

Build `easy-diffusion-ui-render` with SDKit or standalone. FastAPI discovers the
standalone executable at `source/UI.cpp/build/bin/easy-diffusion-ui-render`,
serves styles from `Pages/assets` at `/cpp-ui/assets`, and serves browser glue
from `Pages/src/Plugin/plugin_scripts` at `/cpp-ui/scripts`. You can also set
`SD_UI_CPP_RENDERER` to an executable path. The Logs page displays a bounded
in-memory feed of C++ UI serving events; it does not expose the on-disk
diagnostic log. The Console page captures browser errors and rejected promises.
The original frontend remains available at `/legacy`; changes to its editor
do not automatically add controls to the C++ frontend.

## Advanced Inpaint

Both frontends expose color-weighted inpainting. In the legacy UI at `/legacy`,
use **Image Loader → Advanced Inpaint** or **Canvas → Advanced Inpaint**. The
legacy Canvas tab includes separate Draw, Inpaint, and Advanced Inpaint panes;
opening one deactivates the other panes' keyboard and pointer handling.

Use **Main → Image Loader → Advanced Inpaint**, or open **Canvas** and choose
**Advanced Inpaint** beside **Draw** and **Inpaint**. Load a source image or
start with the blank canvas, select a palette color, and paint guidance strokes.
Brush opacity weights the color hint, rather than committing paint to the
source. Higher opacity supplies stronger color conditioning. The final result
still depends on the checkpoint, prompt, and prompt strength.

Draw, Inpaint, and Advanced Inpaint are independent influences. Draw can edit
the base image. Ordinary inpaint supplies its own mask, processed first.
Advanced Inpaint supplies a separate RGBA color-guidance image, processed
second. It does not replace the ordinary mask or paint into the source.
Both influences are used in **one generation**, not two diffusion passes.

The backend preserves the original request inputs, combines the ordinary mask
with the colored regions for its effective edit mask, and prepares temporary
alpha-weighted color conditioning for the model. It does not paint over the
generated output. This backend uses starting-image conditioning, so color
guidance requires prompt strength strictly between 0 and 1; full-noise strength
1 is rejected rather than silently dropping the hints. Exact final colors are
not guaranteed by every checkpoint.

Both Main image loaders have separate **Draw**, **Inpaint**, and
**Advanced Inpaint** preview tabs. They show the Draw view, ordinary mask
overlay, and color-guidance overlay. Selecting a preview does not change the
generation request. Ordinary inpaint and color guidance have independent
enable controls; disabling one does not discard the other layer.
Overlay previews are reduced for display without resizing the saved inputs.

The image request carries `init_image`, the ordinary `init_image_mask`
(legacy alias `mask`), and the independent `inpaint_color_guide`. If both mask
names are supplied, the canonical field wins, including an explicit null.
The original and color-guidance images must have matching source dimensions.

In the C++ UI, **Use on Main** saves the workspace in this browser's IndexedDB
and returns to Main; it does not start generation. Main's
**Use canvas image and mask** checkbox controls whether the next generation
uses it. When enabled, it takes
precedence over the optional img2img sequence source. Output resolution follows
the Main Image Size controls; source and mask share their original dimensions.

Undo/redo and Clear operate on the current mode. Undo history lasts only for
the current editor visit. Cancel discards changes made during that visit.
The applied workspace persists across reloads on the same browser origin;
**Clear canvas input** removes it, without deleting server images or files.
The C++ editor's source limits are 32 MB and 16 million pixels. In the legacy
editor, **Save** applies that editor's influence to Main. Advanced Save keeps
the original and ordinary mask intact and stores the RGBA color guidance.

### Brush controls in both UIs

Drawing brushes start at **1 px** on load and allow a **1 px minimum**.
Magic Wand Sensitivity is a slider from 0 to 255; zero selects exact RGB
matches and higher values include a wider color range. Selection Buffer is a
slider from 0 to 64 pixels. It expands the contiguous selection outward by
that many pixels. Both frontends use the shared `editor-mask-tools.js`
selection algorithm. Changing either slider affects the next magic-wand click.

Advanced Inpaint retains opacity in its RGBA guidance, including black color
hints. Ordinary masks retain their existing grayscale interpretation.

Renderer discovery respects `SD_UI_CPP_RENDERER` first, then the standalone
UI build, and only then older backend-bundled renderer candidates. This avoids
losing new Canvas controls when launching without an environment override.
