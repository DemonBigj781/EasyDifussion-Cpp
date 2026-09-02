# API runtime test applications

This tree contains small standalone applications for the public handlers
implemented in ../API.cpp. Each application validates the complete
definition -> translation -> Common -> Library route.

Build both applications:

    make

Build and run one backend:

    make run-cpu
    make run-cuda

Build and run the model-byte lifecycle applications:

    make run-load-cpu
    make run-load-cuda
    make run-unload-cpu
    make run-unload-cuda

The Load applications validate empty-input rejection, owned backend allocation,
payload transfer, and payload integrity. The Unload applications validate
release, normalized handle reset, and double-unload rejection.

All four lifecycle applications read MODELS/lifecycle-model.fixture by default.
Pass a different model or binary payload as the first application argument to
exercise a larger local fixture.

The CUDA application requires CUDA development headers, libcudart, a loaded
NVIDIA driver, and access to an NVIDIA device. A restricted container may
compile it successfully but must be granted device access before it can run.

The compiler-only GitHub workflows do not execute these runtime applications.
