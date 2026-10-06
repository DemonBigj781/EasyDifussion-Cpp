import ast
import os
import pathlib
import types
import unittest


class NativeBackendSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = pathlib.Path(__file__).resolve().parents[1]

    def test_platform_picker_exposes_cuda_sycl_and_vulkan(self):
        parameters = (self.root / "ui/media/js/parameters.js").read_text(encoding="utf-8")
        self.assertIn('{ value: "cuda", label: "NVIDIA CUDA" }', parameters)
        self.assertIn('{ value: "cuda-vulkan", label: "NVIDIA CUDA + Vulkan (mixed GPUs, local build)" }', parameters)
        self.assertIn('{ value: "sycl", label: "Intel oneAPI / SYCL (local build)" }', parameters)
        self.assertIn('{ value: "vulkan", label: "Vulkan (experimental)" }', parameters)

    def test_direct_build_supports_sycl_toolchain(self):
        build = (self.root / "source/Build.sh").read_text(encoding="utf-8")
        self.assertIn("--sycl)", build)
        self.assertIn("-DSD_SYCL=\"$USE_SYCL\"", build)
        self.assertIn("-DCMAKE_CXX_COMPILER=icpx", build)
        self.assertIn("-DGGML_SYCL_F16=ON", build)
        self.assertIn("--vulkan)", build)
        self.assertIn("--cuda-vulkan)", build)
        self.assertIn('-DSD_VULKAN="$USE_VULKAN"', build)
        installer = (self.root / "install.sh").read_text(encoding="utf-8")
        self.assertIn("--vulkan)", installer)
        self.assertIn("--cuda-vulkan)", installer)
        self.assertIn('-DSD_VULKAN="$VULKAN_ENABLED"', installer)

    def test_hybrid_backend_uses_the_nvidia_cuda_variant(self):
        launcher = (self.root / "ui/easydiffusion/backends/sdkit3.py").read_text(encoding="utf-8")
        self.assertIn('platform_name in ("cuda", "cuda-vulkan")', launcher)

    def test_explicit_backend_variant_does_not_probe_hardware(self):
        launcher = (self.root / "ui/easydiffusion/backends/sdkit3.py").read_text(encoding="utf-8")
        self.assertNotIn('backend_config.get("variant", get_variant_name(platform_name))', launcher)
        self.assertIn('variant_name = backend_config.get("variant")', launcher)
        self.assertIn("if not variant_name:", launcher)

    def test_cuda_variant_is_ignored_for_non_cuda_targets(self):
        launcher_path = self.root / "ui/easydiffusion/backends/sdkit3.py"
        source = launcher_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(launcher_path))
        get_target_node = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_target"
        )

        config = {}
        detected_platform = "rocm"

        namespace = {
            "OS_NAME": "Linux",
            "getConfig": lambda: config,
            "get_platform_name": lambda: detected_platform,
            "get_variant_name": lambda platform_name: "sm90" if platform_name in ("cuda", "cuda-vulkan") else "any",
            "os": os,
            "platform": types.SimpleNamespace(machine=lambda: "x86_64"),
        }
        exec(compile(ast.Module(body=[get_target_node], type_ignores=[]), str(launcher_path), "exec"), namespace)
        get_target = namespace["get_target"]

        cases = (
            ("sycl", "linux-x64-sycl-any"),
            ("vulkan", "linux-x64-vulkan-any"),
            ("rocm", "linux-x64-rocm-any"),
            ("auto-vulkan", "linux-x64-vulkan-any"),
            ("auto", "linux-x64-rocm-any"),
            ("cuda", "linux-x64-cuda-sm86"),
            ("cuda-vulkan", "linux-x64-cuda-vulkan-sm86"),
            ("auto-cuda", "linux-x64-cuda-sm86"),
        )
        for configured_platform, expected_target in cases:
            with self.subTest(platform=configured_platform):
                config = {
                    "backend_config": {
                        "platform": configured_platform,
                        "variant": "sm86",
                    }
                }
                self.assertEqual(get_target(), expected_target)

    def test_native_server_exposes_runtime_backend_selection(self):
        main = (self.root / "source/sdkit3-port-source/src/main.cpp").read_text(encoding="utf-8")
        generator = (self.root / "source/sdkit3-port-source/src/image_generator.cpp").read_text(encoding="utf-8")
        self.assertIn('"  --backend <name>', main)
        self.assertIn('"  --devices <i,j,...>', main)
        self.assertIn('backend != "vulkan"', main)
        self.assertIn("server_params.compute_backend = compute_backend_spec", main)
        self.assertIn("std::string backend = effective_compute_backend", generator)
        self.assertIn("merge_backend_assignments(compute_backend_, request_compute_backend)", generator)
        self.assertIn('return defaults + "," + request_overrides;', generator)

    def test_sycl_launcher_loads_the_oneapi_runtime_environment(self):
        launcher = (self.root / "ui/easydiffusion/backends/sdkit3.py").read_text(encoding="utf-8")
        self.assertIn("def get_backend_environment(backend_dir):", launcher)
        self.assertIn('environment.get("ONEAPI_SETVARS", "/opt/intel/oneapi/setvars.sh")', launcher)
        self.assertIn("env=environment", launcher)

    def test_wide_settings_editors_use_full_width_rows(self):
        styles = (self.root / "ui/media/css/main.css").read_text(encoding="utf-8")
        self.assertIn('[data-setting-id="backend_commandline_args"],', styles)
        self.assertIn('[data-setting-id="directories"] {', styles)
        self.assertIn("flex: 0 0 100%;", styles)
        self.assertRegex(styles, r"\.model-directories-editor\s*\{[^}]*min-width:\s*0;[^}]*width:\s*100%;")

    def test_per_module_device_controls_are_in_system_settings(self):
        routing = (self.root / "ui/plugins/ui/image_plugin/native-device-routing.plugin.js").read_text(
            encoding="utf-8"
        )
        plugins = (self.root / "ui/media/js/plugins.js").read_text(encoding="utf-8")
        video = (self.root / "ui/plugins/ui/video_plugin/native-video.plugin.js").read_text(encoding="utf-8")
        self.assertIn('fetch("/get/backend_devices"', routing)
        for module in (
            "diffusion",
            "te",
            "llm",
            "controlnet",
            "vae_encode",
            "vae_decode",
            "clip_vision",
            "ip_adapter",
            "latent_interposer_encode",
            "latent_interposer_decode",
        ):
            self.assertIn(f'["{module}",', routing)
        self.assertIn('document.getElementById("system-settings-table")', routing)
        self.assertIn('row.dataset.settingId = "native_device_routing"', routing)
        self.assertNotIn('selector: "#editor-settings-entries"', routing)
        self.assertNotIn('selector: "#sdkit3-controlnet-panel', routing)
        self.assertNotIn('selector: "#sdkit3-native-video-panel', routing)
        self.assertIn("native-device-routing.plugin.js", plugins)
        self.assertIn('assignmentFor("video")', video)

    def test_ip_adapter_compatibility_filter_and_native_guard(self):
        dropdown = (self.root / "ui/media/js/searchable-models.js").read_text(encoding="utf-8")
        plugin = (self.root / "ui/plugins/ui/controlnet_plugin/ip-adapter.plugin.js").read_text(encoding="utf-8")
        adapter = (
            self.root
            / "source/sdkit3-port-source/stable-diffusion.cpp/src/model/adapter/ip_adapter.hpp"
        ).read_text(encoding="utf-8")
        self.assertIn("setModelPredicate(predicate)", dropdown)
        self.assertIn("clip.setModelPredicate", plugin)
        self.assertIn("firstAdapterForFamily", plugin)
        self.assertNotIn('"sdxl/ip-adapter_sdxl"', plugin)
        self.assertIn("expected_clip_embedding_dim", adapter)
        self.assertIn("expected_context_dim", adapter)
        self.assertIn("IP-Adapter/CLIP-Vision shape mismatch", adapter)
        stable = (
            self.root
            / "source/sdkit3-port-source/stable-diffusion.cpp/src/stable-diffusion.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("IP-Adapter/checkpoint shape mismatch", stable)

    def test_pure_noise_plugin_is_required_and_palette_only(self):
        plugins = (self.root / "ui/media/js/plugins.js").read_text(encoding="utf-8")
        noise = (self.root / "ui/plugins/ui/image_plugin/pre-noised-image.plugin.js").read_text(encoding="utf-8")
        self.assertIn("pre-noised-image.plugin.js", plugins)
        self.assertIn("for (let index = 0; index < 8; index += 1)", noise)
        self.assertIn('setInitialImageSource(canvas.toDataURL("image/png"), false)', noise)
        self.assertNotIn("prompt", noise.lower())


if __name__ == "__main__":
    unittest.main()
