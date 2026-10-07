# Corrected native CLI regression

This archive preserves one successful local run of the exact direct image command that previously crashed during native Vulkan model loading. The [report](report.json), [stdout](direct.stdout.log), [stderr](direct.stderr.log), [PNG](image.png), [harness](run.py) and matching packaged metadata/helpers are unchanged copies. [FILES.json](FILES.json) records their original source paths, sizes and SHA-256 hashes: **9 files, 548,533 bytes**. This README, the inventory itself and `.gitattributes` are outside that inventory; `* -text` preserves evidence bytes across Git checkouts.

## Artifact and workload

| Field | Verified value |
| --- | --- |
| Source | `1730424270e35ff4a06bd978654c9990cfd5777e`, clean (`dirty=false`) |
| Application bytes | 190,491,826 |
| Application SHA-256 before and after | `7dd513293f53bc39bd7c92e3aaff0522c89f0d543c3390b4dd95e2f590b507b9` |
| SD1.5 Q4_0 model bytes | 1,747,190,784 |
| Model SHA-256 before and after | `c2f6e92f9d08d69cc673a1003528ac8199274b3c0eaec88d5fbefe5af67bd42b` |
| Selected backend / provider / device | `webgpu` / `native` / `WebGPU0` |
| Adapter | llvmpipe, LLVM 20.1.2; software CPU driver |
| Request | “a red apple on a wooden table”; 256×256, 2 steps, seed 42, CFG 7 |
| Execution settings | 2 application threads; `LP_NUM_THREADS=2`; Euler ancestral / discrete scheduler |
| Completion | Exit 0 after **636.491685 seconds** |

The copied [BUILD.json](application/BUILD.json) binds this result to the corrected CI artifact. The complete argv in the report was compared with the [original failed direct command](../development/local-model-crash/direct-report.json): **only the loader executable, application executable and output PNG paths changed**. All model and inference arguments stayed identical. The successful run used a fresh working directory/TMPDIR, `PATH=/usr/bin:/bin`, and the same pinned local `lvp.json` through both `VK_DRIVER_FILES` and `VK_ICD_FILENAMES`. The full environment is recorded; argv equality does not claim identical transient paths or a byte-identical environment.

## Completion and execution proof

Both completed sampling callbacks arrived exactly once and in order: step 1 took 201.242691 seconds and step 2 took 199.877380 seconds. Final output included `IMAGE_INFERENCE PASS`, `provider=native`, `software=1` and `native_loader_opens=1`.

| Counter scope | Graphs | Submissions | Dispatches | Matrix dispatches | Readbacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete generation after context/model loading | 287 | 1,156 | 6,061 | 1,608 | 855 |
| Interval between completed denoising steps | 126 | 498 | 2,582 | 692 | 370 |

The complete generation window includes CLIP, denoising and VAE work, including lazy weight uploads. The sampling interval excludes the first denoising step, CLIP and final VAE decoding; previews were disabled. CPU fallback is permitted and **its fraction is not measured**. These positive counters establish actual native WebGPU execution without claiming every operation ran there.

The independent standard-library [PNG decoder](application/verify_inference.py) checked dimensions, bounded decompression, chunk CRCs, nonconstant color pixels and agreement with the application's pixel statistics. The result is 256×256 RGB, 157,138 bytes, with PNG SHA-256 `b8c6991041659e06275a1071d3092d430637f041c8de7101ceb7c63a1c7f21e0` and decoded-pixel SHA-256 `98633de44e2f02c4e1f63207c82e57a4d270cc6e44da671a7a8a0bce596caaa1`. This two-step image is a completion/plumbing check; **image quality was not assessed**.

## Separation from other evidence

The old `9ff6b99` application, SHA-256 `b5f1ef5e13adf049e8e938d8ab157807c7748fff9a2934080c5d44ae5e5acf6e`, failed the earlier direct CLI run with SIGSEGV. Its original logs and symbolized loader-worker call stack remain in the separate [development archive](../development/README.md). The old direct CLI reached the text-upload completion marker before crashing, so this regression is established by final generation, PNG and exit success—not by upload progress alone.

A preliminary [permission-rejected launch](prelaunch-permission/report.json) never executed the application or model: the extracted loader lacked executable permission, `Popen` raised `PermissionError`, and the report has no application PID, return code or elapsed runtime. Owner executable permission was restored without changing file bytes, followed by the **single actual application/model run** documented here. The original preliminary report is retained separately and is not counted as an inference attempt or application failure.

This is local native CLI evidence using the corrected CI application. It does not substitute for the final native HTTP CI gate; that gate's completed evidence was still pending when this archive was prepared. No physical GPU was available: `hardware_verified=false`, and native Vulkan here means the host software loader/driver path. No application, loader, debug binary, model checkpoint or dependency archive is included. The archived harness records the original workspace layout and imports the matching packaged helpers; it is provenance, not a standalone relocatable test distribution.
