import base64
import io
from pathlib import Path
import unittest
import sys
import tempfile
from types import ModuleType
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))
fake_app = ModuleType("easydiffusion.app")
fake_app.ROOT_DIR = str(ROOT)
fake_app.MODELS_DIR = str(ROOT / "models")
sys.modules.setdefault("easydiffusion.app", fake_app)

from ui.plugins.server.native_image_tools import native_image_tools


def image_data_url(width=4, height=3):
    image = Image.new("RGB", (width, height), (255, 0, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class SamSegmentationContractTests(unittest.TestCase):
    def test_point_prompt_builds_existing_sam_cli_shape(self):
        request = native_image_tools.SAMSegmentationRequest(
            image=image_data_url(),
            model="sam-vit-b/ggml-model-f16.bin",
            prompt_type="point",
            point={"x": 2, "y": 1},
            multimask=False,
        )

        self.assertEqual(
            native_image_tools._sam_prompt_arguments(request, 4, 3),
            ["-sm", "-p", "2,1"],
        )

    def test_box_prompt_requires_ordered_in_bounds_coordinates(self):
        request = native_image_tools.SAMSegmentationRequest(
            image=image_data_url(),
            model="sam-vit-b/ggml-model-f16.bin",
            prompt_type="box",
            box={"x1": 3, "y1": 1, "x2": 1, "y2": 2},
        )

        with self.assertRaisesRegex(ValueError, "box coordinates"):
            native_image_tools._sam_prompt_arguments(request, 4, 3)

    def test_segment_sam_returns_mask_data_url_from_helper_output(self):
        request = native_image_tools.SAMSegmentationRequest(
            image=image_data_url(),
            model="sam-vit-b/ggml-model-f16.bin",
            prompt_type="point",
            point={"x": 2, "y": 1},
            multimask=False,
        )
        mask = Image.new("L", (4, 3), 255)
        output = io.BytesIO()
        mask.save(output, format="PNG")

        def write_mask(command, timeout):
            output_path = command[command.index("--out") + 1]
            Path(str(output_path) + ".png").write_bytes(output.getvalue())

        with mock.patch.object(native_image_tools, "_sam_binary", return_value="/tmp/sam"), \
             mock.patch.object(native_image_tools, "_sam_model_path", return_value="/models/sam.bin"), \
             mock.patch.object(native_image_tools, "_run_sam", side_effect=write_mask) as run_sam:
            result = native_image_tools.segment_sam(request)

        self.assertTrue(result["mask"].startswith("data:image/png;base64,"))
        self.assertEqual((result["width"], result["height"]), (4, 3))
        run_sam.assert_called_once()
        self.assertIn("-p", run_sam.call_args.args[0])

    def test_cli_prefix_outputs_are_cleaned_up_and_missing_masks_reported(self):
        for multimask, suffixes in ((False, [".png"]), (True, ["1.png", "2.png"]), (False, [])):
            with self.subTest(multimask=multimask, suffixes=suffixes):
                request = native_image_tools.SAMSegmentationRequest(
                    image=image_data_url(), model="sam.bin", prompt_type="point",
                    point={"x": 2, "y": 1}, multimask=multimask)
                output = io.BytesIO()
                Image.new("L", (4, 3), 255).save(output, format="PNG")
                paths = []
                def write_masks(command, timeout):
                    prefix = Path(command[command.index("--out") + 1])
                    paths.append(prefix.parent)
                    for suffix in suffixes:
                        Path(str(prefix) + suffix).write_bytes(output.getvalue())
                with mock.patch.object(native_image_tools, "_sam_binary", return_value="/tmp/sam"), \
                     mock.patch.object(native_image_tools, "_sam_model_path", return_value="/models/sam.bin"), \
                     mock.patch.object(native_image_tools, "_run_sam", side_effect=write_masks):
                    if suffixes:
                        result = native_image_tools.segment_sam(request)
                        if multimask:
                            self.assertEqual(len(result["masks"]), 2)
                            self.assertEqual(result["mask"], result["masks"][0])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "produced no mask"):
                            native_image_tools.segment_sam(request)
                self.assertTrue(paths)
                self.assertTrue(all(not path.exists() for path in paths))

    def test_model_format_guard_uses_signature_not_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.bin"
            for header in (b"GGUF", b"PK\x03\x04", b"", b"bad!"):
                path.write_bytes(header)
                with self.subTest(header=header), self.assertRaisesRegex(ValueError, "converted legacy GGML"):
                    native_image_tools._validate_sam_model_format(path)
            path.write_bytes((0x67676D6C).to_bytes(4, "little"))
            native_image_tools._validate_sam_model_format(path)

    def test_default_sdkit_build_includes_sam_helper(self):
        cmake = (ROOT / "source/sdkit3-port-source/CMakeLists.txt").read_text()
        self.assertIn("add_dependencies(sdkit sam)", cmake)

    def test_server_imports_the_implementation_used_by_the_route(self):
        server = (ROOT / "ui/easydiffusion/server.py").read_text(encoding="utf-8")
        implementation = (ROOT / "ui/plugins/server/native_image_tools/native_image_tools.py").read_text(encoding="utf-8")
        self.assertIn("SAMSegmentationRequest", server)
        self.assertIn("segment_sam as native_segment_sam", server)
        self.assertIn("class SAMSegmentationRequest", implementation)
        self.assertIn("def segment_sam", implementation)

    def test_cpp_and_legacy_panels_have_complete_script_wiring(self):
        cpp_options = (ROOT / "source/UI.cpp/Pages/src/Generate/options.cpp").read_text(encoding="utf-8")
        cpp_pages = (ROOT / "source/UI.cpp/Pages/src/pages.cpp").read_text(encoding="utf-8")
        cpp_script_path = ROOT / "source/UI.cpp/Pages/src/Plugin/plugin_scripts/sam_segmentation.js"
        cpp_script = cpp_script_path.read_text(encoding="utf-8")
        legacy = (ROOT / "ui/plugins/ui/image_plugin/native-image-tools.plugin.js").read_text(encoding="utf-8")
        for control in ("sam_image", "sam_model", "sam_prompt_type", "sam_generate_mask", "sam_mask_preview"):
            self.assertIn(control, cpp_options)
        self.assertIn('/cpp-ui/scripts/sam_segmentation.js', cpp_pages)
        self.assertIn("/sam/segment", cpp_script)
        self.assertIn("/sam/segment", legacy)
        self.assertIn("native-sam-generate", legacy)


if __name__ == "__main__":
    unittest.main()
