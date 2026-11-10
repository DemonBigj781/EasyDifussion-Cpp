import ast
import functools
import glob
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


class RendererSelectionTests(unittest.TestCase):
    def test_override_then_current_ui_then_bundled_fallback(self):
        source = Path(__file__).resolve().parents[1] / 'ui/easydiffusion/server.py'
        function = next(node for node in ast.walk(ast.parse(source.read_text()))
                        if isinstance(node, ast.FunctionDef) and node.name == 'find_cpp_ui_renderer')
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SD_UI_CPP_RENDERER', None)
            root=Path(directory)
            standalone=root/'source/UI.cpp/build/bin/easy-diffusion-ui-render'
            bundled=root/'source/sdkit3-port-source/build/old-backend/bin/easy-diffusion-ui-render'
            for file in (standalone,bundled):
                file.parent.mkdir(parents=True,exist_ok=True);file.write_text('test');file.chmod(0o755)
            namespace={'os':os,'glob':glob,'functools':functools,'app':types.SimpleNamespace(ROOT_DIR=directory)}
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
            choose=namespace['find_cpp_ui_renderer']
            self.assertEqual(choose(),str(standalone))
            os.environ['SD_UI_CPP_RENDERER']=str(bundled);choose.cache_clear()
            self.assertEqual(choose(),str(bundled))
            os.environ.pop('SD_UI_CPP_RENDERER');standalone.unlink();choose.cache_clear()
            self.assertEqual(choose(),str(bundled))
            bundled.unlink();choose.cache_clear();self.assertIsNone(choose())

if __name__=='__main__': unittest.main()
