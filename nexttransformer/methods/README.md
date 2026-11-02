# Method plugins

Each contributed method is one directory here, `nexttransformer/methods/<name>/`, with a `method.json`
declaration and a Python package that implements its hooks. A plugin needs no change to shared files:
the registry discovers it, and its cost proof fingerprints only its own package and the shared core, so
adding it never invalidates another method's proof.

Read the [plugin guide](../../docs/guide/PLUGINS.md) for the declaration, the hooks, the cost rules and the
pull-request checklist. The test plugins in [`tests/plugins/`](../../tests/plugins) show a mixer, a
feed-forward network, and a whole-model plugin with two training stages.
