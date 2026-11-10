import unittest

from easydiffusion import app


class LegacyPluginCatalogTests(unittest.TestCase):
    def test_nested_native_video_plugin_is_discoverable(self):
        self.assertIn(
            "/plugins/core/video_plugin/native-video.plugin.js",
            app.getUIPlugins(),
        )


if __name__ == "__main__":
    unittest.main()
