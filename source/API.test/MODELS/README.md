# API test model fixtures

The lifecycle applications use lifecycle-model.fixture by default. It is a
small opaque byte fixture, not a parsed Stable Diffusion model.

Pass another model or binary payload as the first application argument:

    build/load-cpu MODELS/example-model.bin
    build/load-cuda MODELS/example-model.bin

Large local model files should remain outside version control.
