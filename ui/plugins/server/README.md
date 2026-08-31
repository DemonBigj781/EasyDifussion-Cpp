# Bundled server plugins

Application-owned Python server integrations live here. Core services can be
normal importable packages (for example `tipo`), while lifecycle plugins loaded
automatically by `app.load_server_plugins()` use a `*_plugin.py` module or a
`*_plugin` package. User-installed lifecycle plugins remain in the
repository-level `plugins/server` directory and take precedence over a bundled
plugin with the same module name.
