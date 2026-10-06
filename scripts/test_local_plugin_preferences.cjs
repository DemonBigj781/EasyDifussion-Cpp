const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = path.join(__dirname, "../ui/media/js/local-plugin-preferences.js");
assert.ok(fs.existsSync(source), "A shared plugin preference/catalog implementation is required");
const storage = new Map(), events = [];
const attributes = new Set();
const window = {addEventListener() {}, dispatchEvent(event) { events.push(event); }};
const context = vm.createContext({window, console, Set, Map,
    document: {querySelectorAll: () => [],
        createElement: () => ({}), head: {append() {}},
        documentElement: {toggleAttribute(name, force) { if (force) attributes.add(name); else attributes.delete(name); }}},
    localStorage: {getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value)},
    CustomEvent: class {constructor(type, options) {this.type = type; this.detail = options?.detail;}},
});
vm.runInContext(fs.readFileSync(source, "utf8"), context);
const preferences = window.LocalPluginPreferences;
assert.equal(preferences.storageKey, "easy-diffusion-enabled-local-plugins-v1");
assert.equal(preferences.catalog.length, 26);
assert.equal(new Set(preferences.catalog.map(plugin => plugin.id)).size, 26);
assert.equal(preferences.isEnabled("prompt-assist"), false);
assert.deepEqual([...preferences.getEnabled()].sort(), ["image-modifiers", "perchance-gallery", "perchance-image", "perchance-text"]);
assert.equal(attributes.has("data-image-modifiers-hidden"), false);
preferences.setEnabled("image-modifiers", false);
assert.equal(attributes.has("data-image-modifiers-hidden"), true);
assert.equal(preferences.isEnabled("image-modifiers"), false);
preferences.setEnabled("image-modifiers", true);
assert.equal(attributes.has("data-image-modifiers-hidden"), false);
preferences.saveEnabled(new Set());
assert.equal(preferences.getEnabled().size, 0, "Explicitly disabling all plugins must persist");
preferences.setEnabled("toggle-spellcheck", true);
assert.ok(preferences.isEnabled("toggle-spellcheck"));
preferences.setEnabled("toggle-spellcheck", false);
assert.equal(preferences.isEnabled("toggle-spellcheck"), false);
assert.throws(() => preferences.setEnabled("unknown-plugin", true), /Unknown plugin/);
storage.set(preferences.storageKey, JSON.stringify(["animate", "not-installed"]));
storage.set(preferences.defaultsVersionKey, "1");
assert.deepEqual([...preferences.getEnabled()].sort(), ["animate", "image-modifiers", "perchance-gallery", "perchance-image", "perchance-text"]);
assert.equal(storage.get(preferences.defaultsVersionKey), "3");
storage.set(preferences.storageKey, JSON.stringify([]));
storage.set(preferences.defaultsVersionKey, "2");
assert.deepEqual([...preferences.getEnabled()], ["image-modifiers"], "Migration preserves disabled Perchance plugins");
preferences.setEnabled("image-modifiers", false);
assert.equal(preferences.getEnabled().size, 0, "Reloading preferences must not re-enable hidden modifiers");
assert.ok(events.some(event => event.type === "local-plugin-preferences-changed"));
console.log("PASS: shared 26-entry catalog, opt-in prompt assistance, modifier visibility, defaults migration, and explicit disable");
