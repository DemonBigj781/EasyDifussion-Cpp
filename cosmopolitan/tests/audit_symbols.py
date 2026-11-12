#!/usr/bin/env python3
"""Check that the application build used one GGML implementation."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


def compile_commands_path(out):
    """Select only the compile database belonging to the packaged build.

    Legacy builds use build/. Source-runtime builds are isolated by their
    foundation selection identity; never fall back to a stale legacy build.
    """
    manifest = out / "LINK.json"
    if not manifest.exists():
        return out / "build/compile_commands.json"
    metadata = json.loads(manifest.read_text())
    if not isinstance(metadata, dict):
        raise RuntimeError("LINK.json must contain a metadata object")
    if "foundation_handoff" in metadata:
        handoff = metadata["foundation_handoff"]
        identity = handoff.get("selection_identity_sha256") if isinstance(handoff, dict) else None
        if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise RuntimeError("LINK.json foundation_handoff requires a valid selection_identity_sha256")
        return out / ("build-source-" + identity) / "compile_commands.json"
    if isinstance(metadata.get("cosmocc"), dict) and metadata["cosmocc"].get("version") == "source-built":
        raise RuntimeError("Source-built LINK.json is missing foundation_handoff")
    return out / "build/compile_commands.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    commands = json.loads(compile_commands_path(out).read_text())
    shared = (out / "source/shared-ggml/src").resolve()
    ggml_sources = []
    for command in commands:
        path = Path(command["file"]).resolve()
        tensor_user = (path.is_relative_to(shared) or "/llama.cpp/src/" in path.as_posix()
                       or "/stable-diffusion.cpp/src/" in path.as_posix()
                       or "/cosmopolitan/src/" in path.as_posix()
                       or "/cosmopolitan/shared-ggml/" in path.as_posix())
        if tensor_user and "GGML_MAX_NAME=160" not in command["command"]:
            raise RuntimeError(f"Tensor layout is not consistent in a GGML consumer: {path}")
        if "ggml" not in str(path):
            continue
        if "/ggml/src/" in path.as_posix():
            raise RuntimeError(f"A second vendored GGML implementation is being compiled: {path}")
        if path.is_relative_to(shared):
            if "GGML_MAX_NAME=160" not in command["command"]:
                raise RuntimeError(f"Shared GGML tensor layout is not explicit: {path}")
            ggml_sources.append(path.relative_to(shared).as_posix())
    if not ggml_sources:
        raise RuntimeError("No authoritative shared GGML sources were compiled")
    webgpu_sources = [name for name in ggml_sources if name.startswith("ggml-webgpu/")]
    if webgpu_sources.count("ggml-webgpu/ggml-webgpu.cpp") != 1:
        raise RuntimeError("Expected the original WebGPU backend from the authoritative shared GGML")
    symbols = subprocess.check_output(["nm", "--defined-only", "--format=posix",
                                       str(out / "easy-diffusion.com.dbg")], text=True)
    names = Counter(line.split()[0] for line in symbols.splitlines() if line.strip())
    required = ("ggml_init", "ggml_new_tensor", "ggml_backend_cpu_init", "ggml_backend_dev_count",
                "cosmo_app_create", "cosmo_app_destroy", "cosmo_app_request", "cosmo_app_run",
                "cosmo_shell_run", "cosmo_llama_service_generate", "cosmo_llama_result_free",
                "llama_model_load_from_file", "new_sd_ctx", "cosmo_train_main", "cosmo_sdkit_main",
                "cosmo_image_generate", "cosmo_image_create_context", "cosmo_image_generate_pixels",
                "generate_image", "free_sd_images", "sd_set_sample_progress_callback",
                "ggml_backend_webgpu_init", "ggml_backend_webgpu_reg", "cosmo_webgpu_initialize",
                "cosmo_webgpu_selftest", "cosmo_llama_webgpu_selftest", "cosmo_webgpu_inplace_selftest",
                "wgpuCreateInstance", "wgpuQueueSubmit", "wgpuComputePassEncoderDispatchWorkgroups",
                "wgpuBufferGetConstMappedRange", "cosmo_wgpu_lavapipe_register",
                "lvp_GetInstanceProcAddr", "LLVMCreateMCJITCompilerForModule")
    for name in required:
        if names[name] != 1:
            raise RuntimeError(f"Expected exactly one definition of {name}, found {names[name]}")
    legacy = [name for name in names if name.startswith(("ggml_v2_", "ggml_v3_"))]
    common_sources = [Path(command["file"]).name for command in commands
                      if Path(command["file"]).resolve().parent == (Path(__file__).resolve().parents[1] / "src")]
    for name in ("application.cpp", "application_services.cpp", "application_commands.cpp", "shell.c", "llama_service.cpp"):
        if common_sources.count(name) != 1:
            raise RuntimeError(f"Expected one application source for {name}, found {common_sources.count(name)}")
    if legacy:
        raise RuntimeError("Unexpected alternate GGML implementation symbols")
    report = {"status": "passed", "shared_ggml_sources": sorted(set(ggml_sources)),
              "shared_ggml_webgpu_sources": sorted(set(webgpu_sources)),
              "required_symbol_counts": {name: names[name] for name in required},
              "common_application_sources": sorted(common_sources),
              "legacy_versioned_ggml_symbols": 0,
              "scope": "Source and symbol identity; numerical correctness is tested separately"}
    (out / "SYMBOLS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
