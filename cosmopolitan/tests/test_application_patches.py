#!/usr/bin/env python3
"""Replay application patches and check moved request logic without models/SDKs."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SDKIT = ROOT / "source/sdkit3-port-source"
COSMO = ROOT / "cosmopolitan"
SHARED_PATCH = "0013-shared-application-services.patch"


def patch_targets(patch):
    """Return both sides of unified diffs, using the production -p1 root."""
    result = set()
    for line in patch.read_text().splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        name = line[4:].split("\t", 1)[0]
        if name == "/dev/null":
            continue
        parts = Path(name).parts
        if len(parts) < 2 or parts[0] not in ("a", "b") or ".." in parts:
            raise ValueError("Unexpected patch path: " + name)
        result.add(Path(*parts[1:]))
    return result


def prepare_source_fixture(directory):
    """Use build.py prepare_source's tracked inputs, patch order and cwd roots.

    Only patch targets are copied. In particular, this never hashes or reads
    the model/tokenizer data that a complete source preparation would include.
    The caller owns the temporary directory and may retain it for compilation.
    """
    directory = Path(directory)
    patches = sorted((COSMO / "patches").glob("*.patch")) + sorted(
        (COSMO / "app-patches").glob("*.patch")
    )
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z", "--", "source/llama.cpp",
         "source/sdkit3-port-source", "source/UI.cpp"], cwd=ROOT
    ).decode().split("\0"))
    targets = set()
    for patch in patches:
        base = Path("sdkit3-port-source") if patch.parent.name == "app-patches" else Path()
        targets.update(base / target for target in patch_targets(patch))
    for relative in sorted(targets):
        name = "source/" + relative.as_posix()
        if name not in tracked:
            continue
        original = ROOT / name
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_symlink():
            target.symlink_to(os.readlink(original))
        else:
            if original.stat().st_size > 5 * 1024 * 1024:
                raise ValueError("Unexpected large patch target: " + name)
            shutil.copy2(original, target)
    before_shared = None
    for patch in patches:
        working = directory / "sdkit3-port-source" if patch.parent.name == "app-patches" else directory
        if patch.name == SHARED_PATCH:
            before_shared = (working / "src/server.cpp").read_text()
        result = subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
            cwd=working, text=True, capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(f"{patch.relative_to(ROOT)} failed:\n{result.stdout}{result.stderr}")
    if before_shared is None:
        raise RuntimeError("Shared-services extraction patch was not replayed")
    return before_shared, patches


def section(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def materialize_fixture_headers(directory):
    # Crow's quoted includes resolve beside crow.h, before -I directories.
    # Copy the complete header trees without replacing any patched headers.
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "source/sdkit3-port-source/include",
         "source/sdkit3-port-source/third_party"], cwd=ROOT
    ).decode().split("\0")
    for name in filter(None, names):
        original = ROOT / name
        target = Path(directory) / Path(name).relative_to("source")
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_symlink():
            target.symlink_to(os.readlink(original))
        else:
            shutil.copy2(original, target)


class ApplicationPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="cosmo-application-patches-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before_shared, cls.patches = prepare_source_fixture(cls.directory)
        cls.services = (COSMO / "src/application_services.cpp").read_text()

    def host_compiler(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest("A host C++ compiler is required")
        return compiler

    def test_complete_series_applies_without_fuzz_or_rejects(self):
        self.assertEqual(len(self.patches), 13)
        self.assertFalse(list(self.directory.rglob("*.rej")))

    def test_extracted_request_handlers_preserve_current_source(self):
        for name, following in (("generateVideo", "generateImage"),
                                ("generateImage", "handleProgress"),
                                ("handleProgress", "handleInterrupt")):
            with self.subTest(handler=name):
                original = section(self.before_shared, "crow::response Server::" + name + "(",
                                   "crow::response Server::" + following + "(")
                moved = section(self.services, "crow::response ApplicationServices::" + name + "(",
                                "crow::response ApplicationServices::" + following + "(")
                self.assertEqual(original.replace("Server::", "ApplicationServices::"), moved)

    def test_sampling_and_interposer_helpers_preserve_current_source(self):
        start = "static void parse_native_sampling_optimizations("
        end = "// Round dimension to nearest multiple of 64"
        self.assertEqual(section(self.before_shared, start, end), section(self.services, start, end))

    def test_prompt_preflight_precedes_task_admission_and_error_reset(self):
        for name, following in (("generateVideo", "generateImage"), ("generateImage", "handleProgress")):
            with self.subTest(handler=name):
                body = section(self.services, "crow::response ApplicationServices::" + name + "(",
                               "crow::response ApplicationServices::" + following + "(")
                preflight = body.index("validate_generation_prompts(json_body)")
                self.assertLess(preflight, body.index("GenerationRequestGuard request(*this)"))
                self.assertLess(preflight, body.index("reset_sd_generation_error()"))

    def test_transport_uses_shared_services_instead_of_duplicate_handlers(self):
        server = (self.directory / "sdkit3-port-source/src/server.cpp").read_text()
        self.assertNotIn("Server::generateImage", server)
        self.assertNotIn("Server::generateVideo", server)
        self.assertIn("services_->submitGeneration(kind, pending->body", server)
        self.assertIn('services_->request(COSMO_APP_PROGRESS, req.body)', server)

    def test_shared_services_and_transport_compile_with_patched_headers(self):
        compiler = self.host_compiler()
        materialize_fixture_headers(self.directory)
        staged = self.directory / "sdkit3-port-source"
        vendor = SDKIT / "stable-diffusion.cpp"
        includes = [staged / "include", staged / "third_party", COSMO / "src",
                    COSMO / "backend/include", vendor / "include", vendor / "src",
                    vendor / "ggml/include"]
        command = compiler + ["-std=c++20", "-DGGML_MAX_NAME=160", "-fsyntax-only"]
        command += ["-I" + str(path) for path in includes]
        for source in (COSMO / "src/application_services.cpp", staged / "src/server.cpp"):
            with self.subTest(source=source.name):
                result = subprocess.run(command + [str(source)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_completion_preserves_interruption_and_phase_progress(self):
        compiler = self.host_compiler()
        probe = self.directory / "task-state-probe.cpp"
        probe.write_text('''
#include "task_state.h"
#include <cassert>
int main() {
    TaskStateManager manager;
    for (bool interrupted : {false, true}) {
        const std::string id = interrupted ? "interrupted" : "completed";
        manager.createTask(id);
        manager.updateTaskProgress(id, .25f, "preview", 2, 8);
        manager.updateTaskPhase(id, "sampling", 2, 8);
        if (interrupted) manager.interruptTask(id);
        manager.completeTask(id, {"result"}, "info");
        const auto state = manager.getTaskProgressState(id, true, -1);
        assert(state.completed && state.interrupted == interrupted);
        assert(state.progress == (interrupted ? .25f : 1.0f));
        assert(state.current_step == (interrupted ? 2 : 8));
        assert(state.total_steps == 8);
        assert(state.phase == "finalizing");
        assert(state.phase_current == 8 && state.phase_total == 8);
        assert(state.live_preview == "preview");
        const auto completed = manager.getTaskState(id);
        assert(completed.result_images == std::vector<std::string>{"result"});
        assert(completed.info == "info");
    }
}
''')
        executable = self.directory / "task-state-probe"
        commands = [compiler + ["-std=c++17", "-pthread", "-I" + str(SDKIT / "include"),
                    str(self.directory / "sdkit3-port-source/src/task_state.cpp"),
                    str(probe), "-o", str(executable)], [str(executable)]]
        for command in commands:
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
