#!/usr/bin/env python3
"""Focused tests for AIR file IDs, Anima routing, and the download FIFO."""

from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "ui" / "easydiffusion" / "online_model_browser"
PACKAGE_NAME = "_online_model_browser_tests"


def _load_modules():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(MODULE_ROOT)]
    sys.modules[PACKAGE_NAME] = package

    easy_app = types.SimpleNamespace(ROOT_DIR=str(REPO_ROOT), MODELS_DIR=str(REPO_ROOT / "models"))
    easy_diffusion = types.ModuleType("easydiffusion")
    easy_diffusion.app = easy_app
    easy_diffusion.gallery = types.SimpleNamespace()
    sys.modules["easydiffusion"] = easy_diffusion

    easy_utils = types.ModuleType("easydiffusion.utils")
    easy_utils.log = logging.getLogger("online-model-browser-tests")
    sys.modules["easydiffusion.utils"] = easy_utils

    service_name = f"{PACKAGE_NAME}.service"
    service_spec = importlib.util.spec_from_file_location(service_name, MODULE_ROOT / "service.py")
    service = importlib.util.module_from_spec(service_spec)
    sys.modules[service_name] = service
    assert service_spec and service_spec.loader
    service_spec.loader.exec_module(service)

    fastapi = types.ModuleType("fastapi")

    class APIRouter:
        def get(self, _path):
            return lambda function: function

        def post(self, _path):
            return lambda function: function

    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi.APIRouter = APIRouter
    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    sys.modules["fastapi"] = fastapi

    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = object
    sys.modules["fastapi.responses"] = responses

    huggingface_name = f"{PACKAGE_NAME}.huggingface"
    huggingface_spec = importlib.util.spec_from_file_location(
        huggingface_name,
        MODULE_ROOT / "huggingface.py",
    )
    huggingface = importlib.util.module_from_spec(huggingface_spec)
    sys.modules[huggingface_name] = huggingface
    assert huggingface_spec and huggingface_spec.loader
    huggingface_spec.loader.exec_module(huggingface)

    api_name = f"{PACKAGE_NAME}.api"
    api_spec = importlib.util.spec_from_file_location(api_name, MODULE_ROOT / "api.py")
    api = importlib.util.module_from_spec(api_spec)
    sys.modules[api_name] = api
    assert api_spec and api_spec.loader
    api_spec.loader.exec_module(api)
    return service, huggingface, api


service, huggingface, api = _load_modules()


class OnlineModelBrowserTests(unittest.TestCase):
    def test_air_parser_accepts_legacy_and_exact_file_forms(self):
        self.assertEqual(
            service._parse_urn("urn:air:sdxl:checkpoint:civitai:123@456"),
            (123, 456, None, "sdxl", "checkpoint"),
        )
        self.assertEqual(
            service._parse_urn("urn:air:anima:diffusionmodel:civitai:123@456+789"),
            (123, 456, 789, "anima", "diffusionmodel"),
        )

    def test_exact_air_selects_one_file_and_round_trips_its_id(self):
        item = self._anima_model()
        result = service._trim_model_item(item, version_id=456, file_id=789)
        version = result["modelVersion"]
        self.assertEqual(version["selectedFileId"], 789)
        self.assertEqual([file["id"] for file in version["files"]], [789])
        self.assertEqual(
            version["files"][0]["air"],
            "urn:air:anima:diffusionmodel:civitai:123@456+789",
        )

    def test_unknown_air_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Model file 999"):
            service._trim_model_item(self._anima_model(), version_id=456, file_id=999)

    def test_anima_artifacts_have_explicit_install_buckets(self):
        self.assertEqual(service._base_model_bucket("Anima"), "anima")
        self.assertEqual(
            service._model_type_dir("Checkpoint", "Diffusion Model", "Anima"),
            "DiffusionModels",
        )
        self.assertEqual(service._model_type_dir("LORA", "Model", "Anima"), "Lora")
        self.assertEqual(service._model_type_dir("VAE", "VAE", "Anima"), "VAE")

    def test_huggingface_anima_models_use_the_modular_model_directory(self):
        self.assertEqual(
            huggingface._base_model(["base_model:models/anima-base"], "text-to-image"),
            "Anima",
        )
        target = {
            "id": "example/anima-model",
            "type": "text-to-image",
            "modelVersion": {"baseModel": "Anima"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(service, "_models_dir", return_value=Path(temp_dir)):
                destination = huggingface._destination(target, "anima-base.safetensors")
        self.assertTrue(str(destination).endswith("DiffusionModels/anima"))

    def test_download_uses_authoritative_file_metadata_and_anima_path(self):
        item = service._trim_model_item(self._anima_model(), version_id=456)
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(service, "_models_dir", return_value=Path(temp_dir)):
                url, filename, destination = service._pick_download(
                    item,
                    {"id": 789, "name": "spoofed.safetensors", "downloadUrl": "https://example.com/x"},
                )
        self.assertEqual(url, "https://civitai.com/api/download/models/456?type=Model&format=SafeTensor")
        self.assertEqual(filename, "anima.safetensors")
        self.assertTrue(destination.endswith("DiffusionModels/anima"))

    def test_download_scheduler_is_fifo_and_single_worker(self):
        scheduler = api._DownloadScheduler()
        order = []
        active = 0
        maximum_active = 0
        lock = threading.Lock()
        completed = threading.Event()

        def task(value):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            order.append(value)
            with lock:
                active -= 1
            if len(order) == 3:
                completed.set()

        scheduler.enqueue(task, 1)
        scheduler.enqueue(task, 2)
        scheduler.enqueue(task, 3)
        self.assertTrue(completed.wait(2), "download queue did not drain")
        self.assertEqual(order, [1, 2, 3])
        self.assertEqual(maximum_active, 1)

    @staticmethod
    def _anima_model():
        return {
            "id": 123,
            "name": "Anima test",
            "type": "Checkpoint",
            "modelVersions": [
                {
                    "id": 456,
                    "modelId": 123,
                    "name": "v1",
                    "baseModel": "Anima",
                    "air": "urn:air:anima:diffusionmodel:civitai:123@456",
                    "files": [
                        {
                            "id": 789,
                            "name": "anima.safetensors",
                            "type": "Diffusion Model",
                            "downloadUrl": (
                                "https://civitai.com/api/download/models/456"
                                "?type=Model&format=SafeTensor"
                            ),
                            "primary": True,
                        },
                        {
                            "id": 790,
                            "name": "anima.vae.safetensors",
                            "type": "VAE",
                            "downloadUrl": "https://civitai.com/api/download/models/456?type=VAE",
                        },
                    ],
                    "images": [],
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
