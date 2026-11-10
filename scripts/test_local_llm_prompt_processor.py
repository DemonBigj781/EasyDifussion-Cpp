#!/usr/bin/env python3
"""Focused tests for local LLM settings, placement, inference, and wiring."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))

from fastapi import HTTPException
from easydiffusion import llama_server
from easydiffusion.tipo import llama_backend as easydiffusion_tipo_backend
from ui.plugins.server.local_llm import service
from ui.plugins.server.tipo import llama_backend as plugin_tipo_backend


class FakeModel:
    def __init__(self, content="processed prompt"):
        self.content = content
        self.calls = []
        self.closed = False

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"choices": [{"message": {"content": self.content}}]}

    def close(self):
        self.closed = True


class LocalLLMPromptProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.models_root = Path(self.temp.name)
        self.model = self.models_root / "gguf" / "text" / "prompt-model.gguf"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"GGUF-test")
        self.config = {
            "models_dir": str(self.models_root),
            "local_llm": {
                **service.DEFAULTS,
                "model_path": str(self.model),
                "device": "cpu",
                "gpu_layers": 0,
            },
        }

        def get_config():
            return copy.deepcopy(self.config)

        def set_config(config):
            self.config = copy.deepcopy(config)

        self.get_config = mock.patch.object(service.easy_app, "getConfig", side_effect=get_config)
        self.set_config = mock.patch.object(service.easy_app, "setConfig", side_effect=set_config)
        self.get_config.start()
        self.set_config.start()
        service.shutdown()

    def tearDown(self):
        service.shutdown()
        self.set_config.stop()
        self.get_config.stop()
        self.temp.cleanup()

    def test_model_must_be_a_real_gguf_below_models_root(self):
        outside = Path(self.temp.name).parent / "outside-local-llm.gguf"
        outside.write_bytes(b"GGUF-test")
        try:
            with self.assertRaises(HTTPException) as caught:
                service._validate_settings({**self.config["local_llm"], "model_path": str(outside)})
            self.assertEqual(caught.exception.status_code, 403)
        finally:
            outside.unlink(missing_ok=True)

    def test_cpu_placement_forces_zero_gpu_layers(self):
        settings = service._validate_settings(
            {**self.config["local_llm"], "device": "cpu", "gpu_layers": 999}
        )
        fake = FakeModel()
        with mock.patch.object(service, "LlamaServer", return_value=fake) as llama:
            loaded = service._load_model(settings)
        self.assertIs(loaded, fake)
        self.assertEqual(settings["gpu_layers"], 0)
        self.assertEqual(llama.call_args.args[2], 0)

    def test_cuda_configuration_reports_missing_device(self):
        self.config["local_llm"].update({"device": "cuda", "gpu_layers": 999})
        with mock.patch.object(service, "_cuda_available", return_value=False):
            result = service.health()
        self.assertFalse(result["ready"])
        self.assertIn("no NVIDIA device", result["detail"])

    def test_process_uses_chat_template_and_removes_leading_thought_block(self):
        fake = FakeModel("<think>internal notes</think>\ncinematic fox portrait, rim lighting")
        with mock.patch.object(service, "_load_model", return_value=fake):
            result = service.process_prompt(
                service.ProcessRequest(
                    prompt="fox portrait",
                    instruction="Add cinematic lighting.",
                )
            )
        self.assertEqual(result["processed_prompt"], "cinematic fox portrait, rim lighting")
        messages, kwargs = fake.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Add cinematic lighting.", messages[1]["content"])
        self.assertIn("fox portrait", messages[1]["content"])
        self.assertEqual(kwargs["chat_template_kwargs"], {"enable_thinking": False})

    def test_save_persists_validated_settings_and_restarts_loaded_model(self):
        loaded = FakeModel()
        service._current_model.update({"key": ("old",), "instance": loaded})
        result = service.save_settings({"context_length": 8192})
        self.assertEqual(self.config["local_llm"]["context_length"], 8192)
        self.assertEqual(result["context_length"], 8192)
        self.assertTrue(loaded.closed)

    def test_routes_controls_and_scripts_are_wired(self):
        server = (ROOT / "ui/easydiffusion/server.py").read_text(encoding="utf-8")
        panel = (ROOT / "source/UI.cpp/Pages/src/Generate/plugin_tab_inject.cpp").read_text(
            encoding="utf-8"
        )
        pages = (ROOT / "source/UI.cpp/Pages/src/pages.cpp").read_text(encoding="utf-8")
        legacy = (ROOT / "ui/index.html").read_text(encoding="utf-8")
        script = ROOT / "ui/media/js/local-llm.js"
        for route in ("/local-llm/health", "/local-llm/settings", "/local-llm/process"):
            self.assertIn(route, server)
        for control in ("local-llm-instruction", "local-llm-source-prompt", "local-llm-output"):
            self.assertIn(control, panel)
            self.assertIn(control, legacy)
        self.assertIn("local-llm-settings", pages)
        self.assertIn("local-llm-settings", legacy)
        self.assertIn('/media/js/local-llm.js', pages)
        self.assertIn('media/js/local-llm.js', legacy)
        self.assertTrue(script.is_file())

    def test_both_tipo_package_trees_share_the_generalized_backend(self):
        for backend in (easydiffusion_tipo_backend, plugin_tipo_backend):
            self.assertIs(backend.LlamaServer, llama_server.LlamaServer)
            self.assertIs(backend.backend_readiness, llama_server.backend_readiness)
            self.assertIs(backend.server_executable, llama_server.server_executable)

    def test_shared_backend_preserves_tipo_completion_contract(self):
        model = object.__new__(llama_server.LlamaServer)
        model._post = mock.Mock(return_value={"content": "expanded prompt"})

        result = model.create_completion(
            "prompt",
            temperature=0.6,
            top_p=0.9,
            min_p=0.1,
            top_k=20,
            max_tokens=128,
            repeat_penalty=1.1,
            seed=42,
        )

        self.assertEqual(result["choices"], [{"text": "expanded prompt"}])
        endpoint, payload = model._post.call_args.args
        self.assertEqual(endpoint, "/completion")
        self.assertEqual(payload["prompt"], "prompt")
        self.assertEqual(payload["n_predict"], 128)
        self.assertEqual(payload["seed"], 42)
        self.assertFalse(payload["stream"])


if __name__ == "__main__":
    unittest.main()
