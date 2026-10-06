import importlib.util
import pathlib
import tempfile
import types
import unittest
from unittest import mock


def _load_check_modules():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    module_path = repo_root / "scripts" / "check_modules.py"
    spec = importlib.util.spec_from_file_location("check_modules", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBackendConfigEnvOverrides(unittest.TestCase):
    def test_commandline_args_string(self):
        check_modules = _load_check_modules()
        env = {}
        check_modules.apply_backend_config_env_overrides({"COMMANDLINE_ARGS": "--opt-split-attention"}, env=env)
        self.assertEqual(env.get("COMMANDLINE_ARGS"), "--opt-split-attention")

    def test_commandline_args_list(self):
        check_modules = _load_check_modules()
        env = {}
        check_modules.apply_backend_config_env_overrides(
            {"COMMANDLINE_ARGS": ["--opt-split-attention", "--foo=bar"]}, env=env
        )
        self.assertEqual(env.get("COMMANDLINE_ARGS"), "--opt-split-attention --foo=bar")

    def test_safe_linux_pci_probe_reads_ids_without_invoking_lspci(self):
        check_modules = _load_check_modules()
        device_db = types.ModuleType("torchruntime.device_db")
        device_db.get_linux_output = lambda: "original"

        with tempfile.TemporaryDirectory() as temp_dir:
            sysfs_root = pathlib.Path(temp_dir)
            gpu = sysfs_root / "0000:01:00.0"
            gpu.mkdir()
            (gpu / "vendor").write_text("0x10de\n", encoding="ascii")
            (gpu / "device").write_text("0x2504\n", encoding="ascii")

            with mock.patch.object(check_modules.platform, "system", return_value="Linux"), mock.patch.object(
                check_modules.subprocess, "Popen"
            ) as popen, mock.patch.object(check_modules.subprocess, "check_output") as check_output:
                check_modules.install_safe_linux_pci_probe(device_db, sysfs_root)
                self.assertEqual(device_db.get_linux_output(), "0000:01:00.0 [10de:2504]")

            popen.assert_not_called()
            check_output.assert_not_called()

    def test_safe_linux_pci_probe_ignores_unreadable_or_malformed_devices(self):
        check_modules = _load_check_modules()

        with tempfile.TemporaryDirectory() as temp_dir:
            sysfs_root = pathlib.Path(temp_dir)
            incomplete = sysfs_root / "0000:02:00.0"
            incomplete.mkdir()
            (incomplete / "vendor").write_text("0x10de\n", encoding="ascii")
            malformed = sysfs_root / "0000:03:00.0"
            malformed.mkdir()
            (malformed / "vendor").write_text("0x123\n", encoding="ascii")
            (malformed / "device").write_text("0xabcd\n", encoding="ascii")

            self.assertEqual(check_modules.read_linux_pci_output_from_sysfs(sysfs_root), "")

    def test_torchruntime_configure_skips_unneeded_explicit_backend_detection(self):
        check_modules = _load_check_modules()
        device_db = types.ModuleType("torchruntime.device_db")
        device_db.get_linux_output = lambda: "original"
        torchruntime_package = types.ModuleType("torchruntime")
        torchruntime_package.device_db = device_db

        class FakeTorchRuntime:
            @staticmethod
            def configure():
                raise AssertionError("explicit backend startup must not run TorchRuntime PCI detection")

        with mock.patch.dict(
            "sys.modules",
            {"torchruntime": torchruntime_package, "torchruntime.device_db": device_db},
        ), mock.patch.object(check_modules.subprocess, "Popen") as popen, mock.patch.object(
            check_modules.subprocess, "check_output"
        ) as check_output:
            for configured_platform in (
                "cuda",
                "cuda-vulkan",
                "auto-cuda",
                "auto-vulkan",
                "sycl",
                "vulkan",
                "cpu",
            ):
                with self.subTest(platform=configured_platform):
                    check_modules.configure_torchruntime(
                        FakeTorchRuntime,
                        backend_config={"platform": configured_platform},
                    )

        popen.assert_not_called()
        check_output.assert_not_called()

    def test_torchruntime_info_is_skipped_for_every_explicit_backend(self):
        check_modules = _load_check_modules()

        for configured_platform in (
            "cuda",
            "cuda-vulkan",
            "auto-cuda",
            "auto-vulkan",
            "rocm",
            "sycl",
            "vulkan",
            "cpu",
        ):
            with self.subTest(platform=configured_platform):
                self.assertFalse(check_modules.should_run_torchruntime_info({"platform": configured_platform}))

        self.assertTrue(check_modules.should_run_torchruntime_info({"platform": "auto"}))
        self.assertTrue(check_modules.should_run_torchruntime_info({}))


if __name__ == "__main__":
    unittest.main()
