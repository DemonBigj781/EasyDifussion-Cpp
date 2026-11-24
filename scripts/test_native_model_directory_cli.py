"""Exercise model directory argument parsing against the built native sdkit."""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile


def check_cli(binary, runtime):
    module_path = Path(__file__).resolve().parents[1] / "ui/easydiffusion/backend_args.py"
    spec = importlib.util.spec_from_file_location("backend_args", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = dict(os.environ)
    if runtime:
        env["LD_LIBRARY_PATH"] = str(runtime.resolve())
    with tempfile.TemporaryDirectory(prefix="sdkit-directory-cli-") as cwd:
        def run(*args):
            return subprocess.run([str(binary.resolve()), *args], cwd=cwd, env=env, text=True,
                                  capture_output=True, timeout=10)

        help_text = run("--help")
        assert help_text.returncode == 0, help_text.stderr
        documented = set(re.findall(r"^\s*(--[a-z][a-z0-9-]*)\s+<dir>", help_text.stderr, re.M))
        assert documented == set(module.MODEL_DIRECTORY_FLAGS.values()), documented
        assert module.probe_model_directory_flags(str(binary.resolve()), cwd=cwd, env=env) == module.MODEL_DIRECTORY_FLAGS

        for kind, flag in module.MODEL_DIRECTORY_FLAGS.items():
            accepted = run(flag, cwd + "/Model files", "--help")
            assert accepted.returncode == 0, (flag, accepted.stderr)
            # Do not let the next option silently become the directory value.
            missing = run(flag, "--help")
            assert missing.returncode != 0 and f"{flag} requires a directory argument" in missing.stderr, missing.stderr
            old = module.LEGACY_MODEL_DIRECTORY_FLAGS[kind]
            for args in ((old, cwd, "--help"), (f"{old}={cwd}", "--help")):
                retired = run(*args)
                assert retired.returncode != 0 and f"Use {flag} <dir>" in retired.stderr, retired.stderr

        file_option = run("--control-net-sd1-path", cwd + "/file.safetensors", "--help")
        assert file_option.returncode == 0, file_option.stderr
    print(f"Verified {len(module.MODEL_DIRECTORY_FLAGS)} native directory flags and retired spellings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--runtime", type=Path, help="Directory of shared libraries when not bundled next to sdkit")
    args = parser.parse_args()
    check_cli(args.binary, args.runtime)


if __name__ == "__main__":
    main()
