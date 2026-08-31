import pathlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ui"))

from ui.plugins.server.perchance import perchance


class TestPerchanceImageBatch(unittest.IsolatedAsyncioTestCase):
    async def test_amount_is_bounded(self):
        with self.assertRaises(HTTPException) as raised:
            await perchance.generate_image({"prompt": "test", "amount": 21})
        self.assertEqual(raised.exception.status_code, 400)

    async def test_amount_uses_one_native_count_request(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = pathlib.Path(directory)
            first = output_directory / "first.png"
            second = output_directory / "second.png"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            calls = []

            async def run_perchance(arguments, _timeout):
                calls.append(arguments)
                output = [
                    {"path": str(first), "seed": 100, "width": 512, "height": 512},
                    {"path": str(second), "seed": 101, "width": 512, "height": 512},
                ]
                return {"stdout": f"{json.dumps(output)}\n", "stderr": "", "returncode": 0}

            with (
                patch.object(perchance, "_output_directory", return_value=output_directory),
                patch.object(perchance, "_run_perchance", side_effect=run_perchance),
            ):
                result = await perchance.generate_image(
                    {"prompt": "test", "amount": 2, "seed": 100}
                )

            self.assertEqual(result["generated_amount"], 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index("--count") + 1], "2")
            self.assertIn("--json", calls[0])
            self.assertEqual([image["path"] for image in result["images"]], [str(first), str(second)])
            self.assertEqual([image["seed"] for image in result["images"]], [100, 101])
            self.assertFalse(perchance.PERCHANCE_LOCK.locked())


if __name__ == "__main__":
    unittest.main()
