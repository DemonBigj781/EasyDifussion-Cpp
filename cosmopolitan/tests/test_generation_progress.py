"""Exercise production diffusion callbacks with the real task state manager."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SDKIT = ROOT / "source/sdkit3-port-source"


class GenerationProgressTests(unittest.TestCase):
    def test_image_video_and_interrupted_task_phases(self):
        source = (SDKIT / "src/image_generator.cpp").read_text()
        callbacks = source[source.index("// Global callback data structure"):
                           source.index("// Preview callback")]
        fixture = r'''
#include "task_state.h"
#include <cassert>
class ImageGenerator;
#define LOG_INFO(...) ((void)0)
#define LOG_DEBUG(...) ((void)0)
/* CALLBACKS */
int main() {
    TaskStateManager manager;
    g_callback_data = {nullptr, "", &manager, 4, false};
    progress_callback(1, 4, 0.1f, nullptr); // no task: no effect
    for (bool video : {false, true}) {
        manager.createTask("job");
        g_callback_data.task_id = "job";
        g_callback_data.video_generation = video;
        assert(manager.getTaskState("job").phase == "preparing");
        progress_callback(1, 4, 0.1f, nullptr);
        auto state = manager.getTaskProgressState("job", false, -1);
        assert(state.phase == "sampling");
        assert(state.phase_current == 1 && state.phase_total == 4);
        assert(state.current_step == 1 && state.total_steps == 4);
        assert(state.progress == 0.25f && !state.completed);
        video_frame_progress_callback(1, 2, nullptr);
        assert(manager.getTaskState("job").phase == (video ? "decoding" : "sampling"));
        manager.interruptTask("job");
        manager.completeTask("job", {}, "cancelled");
        state = manager.getTaskProgressState("job", false, -1);
        assert(state.interrupted && state.completed && state.progress == 0.25f);
        assert(state.phase == "finalizing");
        manager.clearTask("job");
    }
    // Reusing an identifier must reset the previous terminal state.
    manager.createTask("job");
    g_callback_data.video_generation = false;
    assert(manager.getTaskState("job").phase == "preparing");
    progress_callback(4, 4, 0.1f, nullptr);
    manager.completeTask("job", {"image"}, "ok");
    auto state = manager.getTaskState("job");
    assert(state.completed && !state.interrupted && state.progress == 1.0f);
    assert(state.phase == "finalizing" && state.result_images.size() == 1);
}
'''.replace("/* CALLBACKS */", callbacks)
        with tempfile.TemporaryDirectory(prefix="generation-progress-") as directory:
            cpp = Path(directory) / "fixture.cpp"
            binary = Path(directory) / "fixture"
            cpp.write_text(fixture)
            command = shlex.split(os.environ.get("CXX", "c++")) + [
                "-std=c++17", "-pthread", "-I" + str(SDKIT / "include"),
                str(cpp), str(SDKIT / "src/task_state.cpp"), "-o", str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


if __name__ == "__main__":
    unittest.main()
