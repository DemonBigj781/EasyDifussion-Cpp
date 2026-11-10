"""Build-registration parity, using host link fixtures, not Cosmocc or inference."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SDKIT = ROOT / "source/sdkit3-port-source"
MODULE = SDKIT / "cmake/runtime-adapters.cmake"


class RuntimeAdapterRegistrationTests(unittest.TestCase):
    def test_both_application_builds_register_runtime_adapters(self):
        for path, root in ((SDKIT / "CMakeLists.txt", "CMAKE_CURRENT_SOURCE_DIR"),
                           (ROOT / "cosmopolitan/CMakeLists.txt", "SDKIT_SOURCE")):
            text = path.read_text()
            self.assertIn("runtime-adapters.cmake", text)
            self.assertIn('sdkit_add_runtime_adapters(stable-diffusion "${' + root + '}")', text)

    def build(self, kind):
        if not shutil.which("cmake") or not shutil.which("c++"):
            self.skipTest("Host CMake and C++ compiler required")
        with tempfile.TemporaryDirectory(prefix="sdkit adapter parity ") as directory:
            work = Path(directory)
            # Existing parser is a link fixture here. Actual parser behavior is
            # exercised by the separately built prompt-validation native test.
            (work / "parser.cpp").write_text('''
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    if (text == "bad") throw std::invalid_argument("weight");
    return {{text, text == "overflow" ? std::numeric_limits<float>::infinity() : 1.0f}};
}
''')
            (work / "main.cpp").write_text('''
#include "native_prompt_validation.h"
int main() {
    if (sdkit_validate_prompt_attention("ok", 2) != SDKIT_PROMPT_VALID) return 1;
    if (sdkit_validate_prompt_attention("bad", 3) != SDKIT_PROMPT_INVALID_WEIGHT) return 2;
    if (sdkit_validate_prompt_attention("overflow", 8) != SDKIT_PROMPT_NONFINITE_WEIGHT) return 3;
    if (sdkit_validate_prompt_attention(nullptr, 0) != SDKIT_PROMPT_VALIDATION_FAILED) return 4;
    return 0;
}
''')
            (work / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.16)
project(adapter_parity LANGUAGES CXX)
add_library(stable-diffusion ''' + kind + ''' parser.cpp)
target_compile_features(stable-diffusion PRIVATE cxx_std_17)
target_include_directories(stable-diffusion PRIVATE
    "${SDKIT_ROOT}/stable-diffusion.cpp/include"
    "${SDKIT_ROOT}/stable-diffusion.cpp/src"
    "${SDKIT_ROOT}/stable-diffusion.cpp/ggml/include")
if(BUILD_SHARED_ADAPTER)
    target_compile_definitions(stable-diffusion PRIVATE SD_BUILD_SHARED_LIB SD_BUILD_DLL)
endif()
include("${SDKIT_ROOT}/cmake/runtime-adapters.cmake")
sdkit_add_runtime_adapters(stable-diffusion "${SDKIT_ROOT}")
add_executable(adapter-check main.cpp)
target_include_directories(adapter-check PRIVATE
    "${SDKIT_ROOT}/include" "${SDKIT_ROOT}/stable-diffusion.cpp/include")
target_link_libraries(adapter-check PRIVATE stable-diffusion)
enable_testing()
add_test(NAME adapter-link COMMAND adapter-check)
''')
            commands = [
                ["cmake", "-S", str(work), "-B", str(work / "build"),
                 "-DSDKIT_ROOT=" + str(SDKIT), "-DBUILD_SHARED_ADAPTER=" + ("ON" if kind == "SHARED" else "OFF")],
                ["cmake", "--build", str(work / "build"), "--parallel", "1"],
                ["ctest", "--test-dir", str(work / "build"), "--output-on-failure"],
            ]
            for command in commands:
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_static_runtime_adapter_link(self):
        self.build("STATIC")

    def test_shared_runtime_adapter_link(self):
        self.build("SHARED")


if __name__ == "__main__":
    unittest.main()
