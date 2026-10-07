# Final Windows and Linux inference evidence

These files preserve the completed results from [CI run 37551341258](https://github.com/DemonBigj781/EasyDifussion-Cpp/actions/runs/37551341258), built from clean source commit `6898a4e564c3363d1692f0aed1a2da0395a17a77`. The application SHA-256 is `ad40cdabb2658bd064473012c32ac8f79bb2ad65557528707930529a888a9516`.

The [inference validation record](../../../../COSMOPOLITAN_INFERENCE_VALIDATION.md) explains the results and limitations; the [machine-readable record](../../../../COSMOPOLITAN_INFERENCE_VALIDATION.json) retains the measurements and source identities.

- `application/` contains the exact packaged build metadata, model pin and host verification scripts. The executable and explicit APE loader are distributed through the linked CI artifact, rather than committed here.
- `linux/` and `windows/` contain the unchanged runtime, bootstrap, image and API result files downloaded from that run. The six generated PNGs are included.
- `run.json`, `jobs.json`, `artifacts.json` and the job logs preserve the completed GitHub records. Download verification files bind the ZIP bytes to GitHub artifact digests.
- `application-review.json` records the independent package/source review. Its static archive and symbol evidence comes from the logged build audits; it does not claim a fresh build-host archive rehash.
- `inspect_ci.py` checks saved reports, hashes, PNGs, counters and lifecycle assertions without running inference. `inspection.json` records its successful inspection with the actual downloaded executable present and rehashed.
- `FILES.json` inventories every other file in this directory by size and SHA-256. Raw text and Windows line endings are preserved by `.gitattributes`.

To inspect downloaded artifacts without running the model:

```sh
python3 cosmopolitan/docs/inference-validation/ci/inspect_ci.py \
  --application-dir /path/to/extracted/application \
  --linux-dir cosmopolitan/docs/inference-validation/ci/linux \
  --windows-dir cosmopolitan/docs/inference-validation/ci/windows
```

The CPU and software-WebGPU CLI tests use the full pinned SD 1.5 checkpoint. The HTTP suite tests the CPU backend. The large-model tests are separate from Linux's embedded-fixture chroot test. Software WebGPU runs on the CPU through the embedded Vulkan software device, with CPU fallback permitted and its fraction unmeasured. Few-step PNG checks establish pipeline execution, not image quality. Native API behavior is tested over real HTTP; UI script coverage uses a DOM mock, not browser automation. Windows API cleanup uses forced process termination with expected status 1 after all assertions pass; Linux records orderly SIGTERM status 0.
