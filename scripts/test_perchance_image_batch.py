import pathlib
import json
import os
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
    async def test_gallery_timeout_allows_nested_frame_startup(self):
        self.assertGreaterEqual(perchance.GALLERY_TIMEOUT_SECONDS, 10 * 60)

    async def test_launcher_prefers_extracted_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            project_root = root / "project"
            home = root / "home"
            extracted = home / "bin" / "perchance"
            appimage = home / "AppImages" / "perchance.AppImage"
            extracted.parent.mkdir(parents=True)
            appimage.parent.mkdir(parents=True)
            extracted.write_bytes(b"launcher")
            appimage.write_bytes(b"appimage")
            extracted.chmod(0o755)
            appimage.chmod(0o755)
            with (
                patch.object(perchance, "ED_ROOT", project_root),
                patch.object(perchance.Path, "home", return_value=home),
                patch.object(perchance.easy_app, "getConfig", return_value={}),
            ):
                self.assertEqual(perchance._launcher_path(), extracted)

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

    async def test_recent_images_recovers_latest_perchance_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = pathlib.Path(directory)
            ignored = output_directory / "native-image.jpeg"
            first = output_directory / f"{'1' * 64}.jpeg"
            second = output_directory / f"{'2' * 64}.jpeg"
            third = output_directory / f"{'3' * 64}.jpeg"
            for path in (ignored, first, second, third):
                path.write_bytes(path.name.encode())
            first.touch()
            second.touch()
            third.touch()
            timestamps = {first: 1_000_000_000, second: 2_000_000_000, third: 3_000_000_000}
            for path, timestamp in timestamps.items():
                os.utime(path, ns=(timestamp, timestamp))

            with patch.object(perchance, "_output_directory", return_value=output_directory):
                result = perchance.recent_images(2)

            self.assertEqual(
                [pathlib.Path(image["path"]).name for image in result["images"]],
                [second.name, third.name],
            )
            self.assertEqual(result["generated_amount"], 2)

    async def test_gallery_uses_lazy_local_cache_without_browser_download(self):
        image_id = "a" * 64
        calls = []

        async def run_perchance(arguments, _timeout):
            calls.append(arguments)
            page = {
                "entries": [{
                    "imageId": image_id,
                    "imageUrl": f"https://aigc.uploads.dev/image/{image_id}.jpeg",
                    "prompt": "test",
                }]
            }
            return {"stdout": json.dumps(page), "stderr": "", "returncode": 0}

        with (
            patch.object(perchance, "_run_perchance", side_effect=run_perchance),
            patch.object(perchance, "get_settings", return_value={"channel": "ai-text-to-image-generator"}),
        ):
            result = await perchance.gallery_list({"download": False})

        self.assertNotIn("--download", calls[0])
        self.assertNotIn("--output", calls[0])
        self.assertEqual(
            result["entries"][0]["local_url"],
            f"/perchance/gallery/cache/{image_id}.jpeg",
        )

    async def test_gallery_retries_transient_frame_startup_failure(self):
        image_id = "d" * 64
        page = {
            "entries": [{
                "imageId": image_id,
                "imageUrl": f"https://aigc.uploads.dev/image/{image_id}.jpeg",
                "prompt": "test",
            }]
        }
        run_perchance = unittest.mock.AsyncMock(side_effect=[
            HTTPException(
                status_code=502,
                detail="The official generator did not open its public gallery frame.",
            ),
            {"stdout": json.dumps(page), "stderr": "", "returncode": 0},
        ])

        with (
            patch.object(perchance, "_run_perchance", run_perchance),
            patch.object(perchance.asyncio, "sleep", unittest.mock.AsyncMock()),
            patch.object(perchance, "get_settings", return_value={"channel": "ai-text-to-image-generator"}),
        ):
            result = await perchance.gallery_list({"download": False})

        self.assertEqual(run_perchance.await_count, 2)
        self.assertEqual(result["entries"][0]["imageId"], image_id)

    async def test_gallery_save_downloads_directly_to_output_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            image_id = "b" * 64
            image_path = output / "perchance-gallery" / f"{image_id}.jpeg"
            calls = []

            async def run_perchance(arguments, _timeout):
                calls.append(arguments)
                page = {
                    "entries": [{
                        "imageId": image_id,
                        "imageUrl": f"https://aigc.uploads.dev/image/{image_id}.jpeg",
                        "prompt": "test",
                    }]
                }
                return {"stdout": json.dumps(page), "stderr": "", "returncode": 0}

            def download(_image_url, target):
                self.assertEqual(target, output / "perchance-gallery")
                image_path.parent.mkdir()
                image_path.write_bytes(b"image")
                return image_path

            with (
                patch.object(perchance, "_output_directory", return_value=output),
                patch.object(perchance, "_download_gallery_image", side_effect=download),
                patch.object(perchance, "_run_perchance", side_effect=run_perchance),
                patch.object(perchance, "get_settings", return_value={"channel": "ai-text-to-image-generator"}),
            ):
                result = await perchance.gallery_list({"download": True})

            self.assertNotIn("--download", calls[0])
            self.assertEqual(
                result["entries"][0]["local_url"],
                f"/perchance/file/perchance-gallery/{image_id}.jpeg",
            )

    async def test_gallery_cache_resolution_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = pathlib.Path(directory) / "cache"
            cache.mkdir()
            image = cache / "safe.png"
            image.write_bytes(b"image")
            with patch.object(perchance, "_gallery_cache_directory", return_value=cache):
                self.assertEqual(perchance.resolve_gallery_cache_file("safe.png"), image)
                with self.assertRaises(HTTPException) as raised:
                    perchance.resolve_gallery_cache_file("../outside.png")
            self.assertEqual(raised.exception.status_code, 403)

    async def test_gallery_cache_fetches_a_missing_trusted_image(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = pathlib.Path(directory) / "cache"
            cache.mkdir()
            image_id = "c" * 64
            expected = cache / f"{image_id}.webp"

            def download(image_url, target):
                self.assertEqual(
                    image_url,
                    f"https://aigc.uploads.dev/image/{image_id}.webp",
                )
                self.assertEqual(target, cache)
                expected.write_bytes(b"image")
                return expected

            with (
                patch.object(perchance, "_gallery_cache_directory", return_value=cache),
                patch.object(perchance, "_download_gallery_image", side_effect=download),
            ):
                self.assertEqual(
                    perchance.resolve_gallery_cache_file(f"{image_id}.webp"),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
