# Pages

Store page definitions and page-level composition here. `src/pages.cpp` holds
the navigation catalog and C++ page builders. Pages assemble shared components from
`../Shared_Core` and feature-owned components, while leaving reusable
primitives in their owning modules.

The catalog provides Main, Canvas, Perchance Text/Image/Gallery, Image Gallery,
Dataset Gallery, Tagging, Training, Model Downloading, Settings, GPU Config,
Plugin Config, Stats, Logs, and Console destinations. Perchance and Galleries
appear as grouped primary navigation menus. System pages are in a separate
navigation row. The FastAPI host serves the C++ Main page at `/` when the renderer is available;
other C++ pages use the `/cpp-ui` namespace. The page builders provide the
navigation and page structure. Main generation is split into `Generate/Main.cpp`
and feature modules under `Generate/`; its browser bridge loads checkpoints,
submits `/render`, handles stop/progress, and displays returned images. LoRA and
ControlNet panels contribute request options through the Generate plugin
injection point. Other page interactions are being connected incrementally.
Dataset Gallery keeps processed images beside the selected-for-dataset column
and reserves the hidden pre-filter area for duplicate detection.

Build `easy-diffusion-ui-render` with SDKit or standalone. FastAPI discovers the
standalone executable at `source/UI.cpp/build/bin/easy-diffusion-ui-render`,
serves styles from `Pages/assets` at `/cpp-ui/assets`, and serves browser glue
from `Pages/src/Plugin/plugin_scripts` at `/cpp-ui/scripts`. You can also set
`SD_UI_CPP_RENDERER` to an executable path. The Logs page displays a bounded
in-memory feed of C++ UI serving events; it does not expose the on-disk
diagnostic log. The Console page captures browser errors and rejected promises.
The Logs page displays a bounded in-memory feed of C++ UI serving events; it
does not expose the on-disk diagnostic log. The Console page captures browser
errors and rejected promises. The existing `/` page remains available unchanged.
