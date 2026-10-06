import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from fastapi import HTTPException

package = types.ModuleType("easydiffusion")
setattr(package, "app", types.SimpleNamespace(getConfig=lambda: {"kiosk_mode": True}))
path = Path(__file__).resolve().parents[1] / "ui/easydiffusion/kiosk.py"
spec = importlib.util.spec_from_file_location("kiosk_under_test", path)
assert spec is not None and spec.loader is not None
kiosk = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"easydiffusion": package}):
    spec.loader.exec_module(kiosk)


class KioskTests(unittest.TestCase):
    def test_base_generation_stays_allowed(self):
        kiosk.validate_render({"use_stable_diffusion_model": "Anima/anima-base-v1.0"})

    def test_adapter_cannot_smuggle_bundled_lora_into_kiosk(self):
        for name, value in [("ip_adapter_model", "renamed"), ("ip_adapter_image", "data:image/png;base64,AA=="),
                            ("ip_adapter_clip_vision", "siglip2")]:
            with self.subTest(name=name):
                with self.assertRaises(HTTPException) as result:
                    kiosk.validate_render({"use_stable_diffusion_model": "Anima/anima-base-v1.0", name: value})
                self.assertEqual(result.exception.status_code, 403)

    def test_adapter_remains_available_outside_kiosk(self):
        with patch.object(kiosk, "enabled", return_value=False):
            kiosk.validate_render({"use_stable_diffusion_model": "Anima/anima-base-v1.0", "ip_adapter_model": "renamed"})


if __name__ == "__main__":
    unittest.main()
