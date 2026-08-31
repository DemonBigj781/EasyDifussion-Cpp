import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ui"))

from ui.plugins.server.native_image_tools import native_image_tools


class TestNativeImageTools(unittest.TestCase):
    def test_vision_process_uses_active_python_libtorch(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = pathlib.Path(directory)
            torch_lib = package_root / "torch" / "lib"
            torch_lib.mkdir(parents=True)
            (torch_lib / "libtorch.so").write_bytes(b"test")
            with (
                patch.object(
                    native_image_tools.sysconfig,
                    "get_paths",
                    return_value={"platlib": str(package_root), "purelib": str(package_root)},
                ),
                patch.dict(os.environ, {"LD_LIBRARY_PATH": "/existing/runtime"}, clear=False),
            ):
                environment = native_image_tools._runtime_environment(["/tmp/sdkit-vision"])

            self.assertEqual(
                environment["LD_LIBRARY_PATH"].split(os.pathsep),
                [str(torch_lib), "/existing/runtime"],
            )

    def test_other_helpers_do_not_require_libtorch(self):
        with patch.object(
            native_image_tools,
            "_torch_library_directory",
            side_effect=AssertionError("LibTorch lookup should not run"),
        ):
            environment = native_image_tools._runtime_environment(["/tmp/sdkit-image-tools"])
        self.assertIsInstance(environment, dict)


if __name__ == "__main__":
    unittest.main()
