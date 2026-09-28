# Unified UI module

`source/UI.cpp/Shared_Core` is the shared C++ presentation layer. It provides escaped HTML nodes,
complete page assembly, and a registry for reusable components. Feature code can
build pages from shared components without owning HTML serialization.

## Current contract

- `easy_diffusion::ui::Node` represents text or an HTML element. Text and
  attribute values are escaped during rendering. Inline scripts, styles,
  event-handler attributes, and unsafe URL schemes are rejected.
- `easy_diffusion::ui::Page` assembles a document from a title, language,
  stylesheet and script URLs, and a body node. Scripts are linked with
  `defer`; executable code stays in external assets.
- `easy_diffusion::ui::ComponentRegistry` maps stable component names to
  factories that accept string properties and return nodes. This is the shared
  home for common UI components such as buttons, dropdowns, and file selectors.
- `source/UI.cpp/Shared_Core/CMakeLists.txt` builds the standalone `easy-diffusion-ui`
  static library with a C++17 interface.

Feature-specific templates remain with their feature. For example, gallery
templates belong under `ui/plugins/server/gallery/ui`; the unified module is
for reusable rendering primitives and shared components, not a reason to
flatten feature ownership.

## Integration boundary

The current application UI is still served by the Python FastAPI/browser
implementation. This library is not yet wired into that server, SDKit, or the
application build. An adapter must be designed before moving live templates or
routes: it needs to define how C++ page/component output is registered with the
existing server, how static assets are served, and how feature routes pass
typed data into component properties. Until then, keep existing UI behavior in
place and use this library as the implementation target for new shared C++ UI
primitives.

## Minimal use

```cpp
using namespace easy_diffusion::ui;

ComponentRegistry components;
components.register_component("primary-button", [](const Properties& props) {
    const auto label = props.find("label");
    return Node::element("button", {{"class", "primary"}},
                         {Node::text(label == props.end() ? "Submit" : label->second)});
});

Page page("Example");
page.add_stylesheet("/assets/app.css")
    .add_script("/assets/app.js")
    .set_body(Node::element("main", {}, {components.render_component(
        "primary-button", {{"label", "Continue"}})}));
```
