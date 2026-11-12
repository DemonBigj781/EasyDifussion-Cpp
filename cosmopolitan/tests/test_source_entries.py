import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build

class SourceEntryTests(unittest.TestCase):
    def enumerate(self, records):
        with mock.patch.object(build.subprocess, 'check_output', return_value=records):
            return build.application_source_entries(['source'])
    def test_only_disabled_webm_gitlink_is_omitted_and_pinned(self):
        revision = 'a' * 40
        name = 'source/sdkit3-port-source/stable-diffusion.cpp/thirdparty/libwebm'
        data = ('100644 ' + revision + ' 0\tsource/a.cpp\0' +
                '120000 ' + revision + ' 0\tsource/link\0' +
                '160000 ' + revision + ' 0\t' + name + '\0').encode()
        self.assertEqual(self.enumerate(data), (['source/a.cpp', 'source/link'], [(name, revision)]))
        cmake = (Path(build.HERE) / 'CMakeLists.txt').read_text()
        self.assertIn('SD_WEBP SD_WEBM SD_CUDA', cmake)
    def test_unknown_gitlink_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Unreviewed application gitlink'):
            self.enumerate(b'160000 aaaa 0\tsource/unknown\0')
    def test_unmerged_source_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Unmerged application source'):
            self.enumerate(b'100644 aaaa 2\tsource/file\0')
    def test_unknown_file_mode_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Unsupported application source mode'):
            self.enumerate(b'040000 aaaa 0\tsource/directory\0')

if __name__ == '__main__':
    unittest.main()
