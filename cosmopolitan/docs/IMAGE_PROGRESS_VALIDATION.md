# Image sampling phase regression

Ordinary image generation and video generation share the native sample callback. Previously only video updated the task phase, leaving image tasks labelled `preparing` even after sampling began. The callback now updates the sampling phase for both paths. Video frame decoding behavior is unchanged.

`test_generation_progress.py` compiles the actual callbacks with the real task-state manager. It covers image/video sampling, progress polling, video decoding, cancellation, completion, and task-ID reuse. The test fails against the preceding callback implementation.

Local validation: 161 host tests passed with 14 skips for unavailable native build dependencies. CI is required for the complete native, browser, portable, and Windows qualification. This fix does not add a VAE-decoding callback for ordinary images and does not claim GPU or model-quality validation.
