import os
import pathlib
import sys
import tempfile
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ui"))

from ui.plugins.server.native_image_tools import native_image_tools


class TestNativeImageTools(unittest.TestCase):
    @staticmethod
    def image_data_url(mode="RGB", color=(20, 40, 60), size=(16, 12)):
        buffer = BytesIO()
        Image.new(mode, size, color).save(buffer, format="PNG")
        import base64

        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

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

    def test_background_removal_uses_bundled_model_store_and_returns_png(self):
        class FakeSession:
            def get_inputs(self):
                return [SimpleNamespace(name="input.1")]

            def run(self, _outputs, feeds):
                self.test_shape = feeds["input.1"].shape
                mask = np.linspace(0, 1, 320 * 320, dtype=np.float32).reshape(1, 1, 320, 320)
                return [mask]

        session = FakeSession()
        request = native_image_tools.BackgroundRemovalRequest(image=self.image_data_url())
        with patch.object(native_image_tools, "_background_removal_session", return_value=session):
            result = native_image_tools.remove_background(request)
        self.assertEqual(session.test_shape, (1, 3, 320, 320))
        self.assertEqual((result["width"], result["height"]), (16, 12))
        self.assertTrue(result["image"].startswith("data:image/png;base64,"))
        self.assertEqual(
            native_image_tools._source_model_directory("remove_background_onnx"),
            pathlib.Path(native_image_tools.app.ROOT_DIR) / "source/models/remove_background_onnx",
        )

    def test_object_removal_runs_converted_onnx_contract(self):
        mask = self.image_data_url("L", 255)

        class FakeSession:
            def get_inputs(self):
                return [SimpleNamespace(name="input:0")]

            def run(self, _outputs, feeds):
                value = feeds["input:0"]
                self.test_shape = value.shape
                return [np.full((1, 400, 400, 3), 0.5, dtype=np.float32)]

        session = FakeSession()
        request = native_image_tools.ObjectRemovalRequest(
            image=self.image_data_url(),
            mask=mask,
            feather=0,
        )
        with patch.object(native_image_tools, "_object_removal_session", return_value=session):
            result = native_image_tools.remove_object(request)
        self.assertEqual(session.test_shape, (1, 400, 400, 3))
        self.assertEqual((result["width"], result["height"]), (16, 12))
        self.assertEqual(result["model"], "VPanjeta/Deep-Object-Removal")

    def test_object_model_prefers_requested_source_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / "source/models/remove_object_onnx/deep-object-removal-400.onnx"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"onnx")
            with (
                patch.object(native_image_tools.app, "ROOT_DIR", str(root)),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.assertEqual(native_image_tools._object_removal_model(), model)


if __name__ == "__main__":
    unittest.main()
