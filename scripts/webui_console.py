"""Open a Linux developer shell in Easy Diffusion's contained environment."""

import os
import platform
import subprocess
from pathlib import Path


def main() -> None:
    if platform.system() != "Linux":
        raise SystemExit("This local Easy Diffusion fork currently supports Linux only.")

    root_dir = Path(__file__).resolve().parent.parent
    env_dir = root_dir / "installer_files" / "env"
    python = env_dir / "bin" / "python"
    if not python.is_file():
        raise SystemExit(f"Easy Diffusion's contained Python is missing: {python}")

    version = subprocess.check_output(
        [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True,
    ).strip()
    site_packages = env_dir / "lib" / f"python{version}" / "site-packages"

    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PATH"] = os.pathsep.join((str(env_dir / "bin"), env.get("PATH", "")))
    env["PYTHONNOUSERSITE"] = "y"
    env["PYTHONPATH"] = os.pathsep.join((str(root_dir / "ui"), str(site_packages)))
    env["SD_UI_PATH"] = str(root_dir / "ui")
    env["INSTALL_ENV_DIR"] = str(env_dir)

    print("Easy Diffusion developer environment")
    print(f"Root: {root_dir}")
    print(f"Python: {python}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")

    shell = env.get("SHELL") or "/bin/bash"
    os.chdir(root_dir)
    os.execvpe(shell, [shell, "-i"], env)


if __name__ == "__main__":
    main()
